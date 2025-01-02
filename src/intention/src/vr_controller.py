#!/usr/bin/python3

# Basic modules
import os
import sys
import rospy
import roslib
import numpy as np

# TF modules
import tf
from tf.transformations import (
    euler_from_quaternion,
    quaternion_from_euler,
    quaternion_inverse,
    quaternion_multiply,
)

# Messages
from std_msgs.msg import *
from sensor_msgs.msg import *
from geometry_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *

# Custom modules
import rtde_control
import rtde_receive
from enum import Enum


class MetaController:
    def __init__(self):
        # Right Controller Data : Initialize
        self.right_controller_twist = Twist()
        self.right_controller_joy = Joy()
        self.right_controller_joy.buttons = [0, 0, 0, 0]

        # Left Controller Data : Initialize
        self.left_controller_joy = Joy()
        self.left_controller_joy.buttons = [0, 0, 0, 0]

        # 오른쪽 - [A, B, 중지, 검지]
        # 왼쪽 - [X, Y, 중지, 검지]

        # Subscribers
        self.right_controller_twist_sub = rospy.Subscriber(
            "/controller/right/twist", Twist, self.right_controller_callback
        )
        self.right_controller_joy_sub = rospy.Subscriber(
            "/controller/right/joy", Joy, self.right_controller_joy_callback
        )
        self.left_controller_joy_sub = rospy.Subscriber(
            "/controller/left/joy", Joy, self.left_controller_joy_callback
        )

    def right_controller_joy_callback(self, msg: Joy):
        self.right_controller_joy = msg

    def left_controller_joy_callback(self, msg: Joy):
        self.left_controller_joy = msg

    def right_controller_callback(self, msg: Twist):
        self.right_controller_twist = msg

    def get_control_input(self) -> np.array:

        # 오른손 검지 버튼이 눌렸을 때, 속도 명령 리턴
        if self.right_controller_joy.buttons[3] == 1:
            linear_vel = self.right_controller_twist.linear
            return True, np.array(
                [linear_vel.y, -linear_vel.x, linear_vel.z, 0.0, 0.0, 0.0]
            )

        # 오른손 검지 버튼이 눌리지 않았을 때, 정지 명령 부여
        else:
            return False, np.array([0.0] * 6)


class AutoController:
    def __init__(self):
        self.auto_controller_twist_sub = rospy.Subscriber(
            "/auto_controller/twist", Twist, self.auto_controller_callback
        )

        self.auto_controller_twist = Twist()
        self.bayesian_success = False

    def auto_controller_callback(self, msg: Twist):
        self.auto_controller_twist = msg

    def get_control_input(
        self, v: float = 0.15, forward_gain: float = 1.0, lateral_gain: float = 1.0
    ) -> np.array:
        linear_vel = self.auto_controller_twist.linear

        velocity_matrix = np.array(
            [
                -linear_vel.x * forward_gain,
                -linear_vel.y * lateral_gain,
                linear_vel.z * lateral_gain,
                0.0,
                0.0,
                0.0,
            ]
        )

        if np.isclose(np.linalg.norm(velocity_matrix), 0.0):
            # 속도 명령이 0이면 정지 명령 부여
            return False, np.array([0.0] * 6)

        # 정규화된 속도에 대한 제어 입력
        return True, v * velocity_matrix


class URControl:
    class ControlMode(Enum):
        MANUAL = 0
        AUTO = 1
        FORCE_CONTROL = 2
        RESET = 8
        HOMING = 9

    def __init__(self, hz: int = 30):
        self.hz = hz

        IP = "192.168.2.2"

        # RTDE Control and Receive Interface
        self.rtde_c = rtde_control.RTDEControlInterface(IP)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(IP)

        # Controllers
        self.manual_controller = MetaController()
        self.auto_controller = AutoController()

        # Control Mode
        self.control_mode = self.ControlMode.MANUAL
        self.control_mode_pub = rospy.Publisher("/control_mode", UInt16, queue_size=10)

        # Vacuum Gripper Publisher
        self.vacuum_gripper_pub = rospy.Publisher("/vacuum", String, queue_size=10)
        self.vacuum_gripper_state = False

        # Joy Stick
        self.right_controller_joy_sub = rospy.Subscriber(
            "/controller/right/joy", Joy, self.right_controller_joy_callback
        )

        self.baysian_success_sub = rospy.Subscriber(
            "/auto_controller/bayisian_success", Bool, self.bayesian_success_callback
        )

        self.triggered_time = rospy.Time.now()

        # @@@@@@@@@ Subscribers @@@@@@@@@

        # Distance Subscriber : 거리가 임계점보다 가까우면 자동 제어로 전환
        self.distance_sub = rospy.Subscriber(
            "/intention/distance", Float32, self.distance_callback
        )
        self.eef_distance = float("inf")
        self.eef_distance_threshold = 0.2

    def bayesian_success_callback(self, msg: Bool):
        is_success = msg.data

        # 컨트롤 모드가 수동이고, 베이지안 성공 메시지가 True일 때, 자동 제어 모드로 전환
        if self.control_mode == self.ControlMode.MANUAL:
            if is_success:
                self.control_mode = self.ControlMode.AUTO

                rospy.loginfo("Mode Change: Auto - Bayesian Success")

                # 그리퍼를 ON 상태로 변경
                self.vacuum_gripper_state = True
                vacuum_command = "on" if self.vacuum_gripper_state else "off"
                self.vacuum_gripper_pub.publish(vacuum_command)

            elif not is_success and self.eef_distance < self.eef_distance:
                self.control_mode = self.ControlMode.AUTO

                rospy.loginfo("Mode Change: Auto - Distance Threshold")

                # 그리퍼를 ON 상태로 변경
                self.vacuum_gripper_state = True
                vacuum_command = "on" if self.vacuum_gripper_state else "off"
                self.vacuum_gripper_pub.publish(vacuum_command)

    def intersection_length_callback(self, msg: Int32):
        self.intersection_length = msg

    def distance_callback(self, msg: Float32):
        self.eef_distance = msg.data

    def right_controller_joy_callback(self, msg: Joy):
        # 오른쪽 - [A, B, 중지, 검지]
        if len(msg.buttons) != 4:
            # 예외 처리
            rospy.logwarn("Invalid Right Controller Joy Message")
            return None

        # A 버튼이 눌렸을 때
        if msg.buttons[0] == 1:
            pass

        # B 버튼이 눌렸을 때
        if msg.buttons[1] == 1:
            # 그리퍼를 OFF 전환하고 상태를 Homing 상태로 변경
            self.vacuum_gripper_state = False
            vacuum_command = "on" if self.vacuum_gripper_state else "off"
            self.vacuum_gripper_pub.publish(vacuum_command)

            rospy.loginfo("Mode Change: Homeing")

            self.control_mode = self.ControlMode.HOMING
            self.triggered_time = rospy.Time.now()

        # 중지 버튼이 눌렸을 때
        if msg.buttons[2] == 1:
            # 강제 수동 제어 상태로 변경

            rospy.loginfo("Mode Change: Force Control")

            self.control_mode = self.ControlMode.FORCE_CONTROL

        # 중지 버튼이 눌리지 않았을 때, 강제 수동 제어 해제
        else:
            if self.control_mode == self.ControlMode.FORCE_CONTROL:

                rospy.loginfo("Mode Change: Manual")

                self.control_mode = self.ControlMode.MANUAL

    def control(self):
        # 리셋 상태일 때, 상태를 1회 발행하고 수동 제어 상태로 변경
        if self.control_mode == self.ControlMode.RESET:

            rospy.loginfo("Mode Change: Manual")

            self.control_mode_pub.publish(UInt16(data=self.control_mode.value))
            self.control_mode = self.ControlMode.MANUAL

        # Homing 상태일 때, 로봇을 초기 위치로 이동(planner에 의해 처리), 이동이 끝나면 RESET 상태로 변경
        elif self.control_mode == self.ControlMode.HOMING:
            current_time = rospy.Time.now()

            if (current_time - self.triggered_time).to_sec() < 1.0:
                # X로 뒤로 0.05m/s 로 2초간 이동
                homing_input = [0.05, 0.0, 0.0, 0.0, 0.0, 0.0]

            else:
                is_running, homing_input = self.auto_controller.get_control_input(v=0.1)

                if is_running:
                    # 속도가 0이 아니라면 명령 수행
                    self.rtde_c.speedL(homing_input, acceleration=0.25)

                else:
                    # 속도가 0이라면 RESET 상태로 변경

                    rospy.loginfo("Mode Change: Reset")

                    self.control_mode = self.ControlMode.RESET

        # 수동, 자동, 강제 수동 제어 상태일 때, 로봇을 제어
        else:
            combined_input = [0.0] * 6  # 명령 초기화

            if (
                self.control_mode == self.ControlMode.MANUAL
                or self.control_mode == self.ControlMode.FORCE_CONTROL
                or self.control_mode == self.ControlMode.HOMING
            ):
                # 수동 제어 상태에서는 메타 컨트롤러로부터 제어 입력을 받음
                _, combined_input = self.manual_controller.get_control_input()

            elif self.control_mode == self.ControlMode.AUTO:
                # 자동 제어 상태에서는 자동 컨트롤러로부터 제어 입력을 받음

                _, combined_input = self.auto_controller.get_control_input(
                    v=0.15, forward_gain=1.0, lateral_gain=2.0
                )

            # 로봇에 속도 명령을 부여
            self.rtde_c.speedL(combined_input, acceleration=0.25)

        # 제어 모드를 발행
        self.control_mode_pub.publish(UInt16(data=self.control_mode.value))


def main():
    rospy.init_node("control_test_node")  # TODO: Add node name

    hz = 20

    robot_control = URControl(hz=hz)

    r = rospy.Rate(hz)  # TODO: Add rate
    while not rospy.is_shutdown():

        robot_control.control()

        r.sleep()


if __name__ == "__main__":
    try:
        main()
        pass
    except rospy.ROSInterruptException as ros_ex:
        rospy.logfatal("ROS Interrupted.")
        rospy.logfatal(ros_ex)
    except Exception as ex:
        rospy.logfatal("Exception occurred.")
        rospy.logfatal(ex)
    finally:
        rospy.loginfo("Shutting down.")
        rospy.signal_shutdown("Shutting down.")
        sys.exit(0)
