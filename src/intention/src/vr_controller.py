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
        # 오른손 검지 버튼이 눌렸을 때만, 속도 명령 부여

        if self.right_controller_joy.buttons[3] == 1:
            linear_vel = self.right_controller_twist.linear
            return np.array([linear_vel.y, -linear_vel.x, linear_vel.z, 0.0, 0.0, 0.0])

        # 오른손 검지 버튼이 눌리지 않았을 때, 정지 명령 부여
        else:
            return np.array([0.0] * 6)


class AutoController:
    def __init__(self):
        self.auto_controller_twist_sub = rospy.Subscriber(
            "/auto_controller/twist", Twist, self.auto_controller_callback
        )

        self.auto_controller_twist = Twist()

    def auto_controller_callback(self, msg: Twist):
        self.auto_controller_twist = msg

    def get_control_input(self, v: float = 0.15) -> np.array:
        linear_vel = self.auto_controller_twist.linear

        forward_gain = 1.0
        lateral_gain = 2.2

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

        # 정규화된 속도에 대한 제어 입력
        return v * velocity_matrix


class URControl:
    class ControlMode(Enum):
        MANUAL = 0
        SEMI = 1  # 안씀
        AUTO = 2
        FORCE_CONTROL = 3

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

        # Joy Stick
        self.right_controller_joy_sub = rospy.Subscriber(
            "/controller/right/joy", Joy, self.right_controller_joy_callback
        )

        # Vacuum Gripper
        self.vacuum_gripper_pub = rospy.Publisher("/vacuum", String, queue_size=10)
        self.vacuum_gripper_state = False

        # Bayisian Filter
        self.bayisian_filter_sub = rospy.Subscriber(
            "/auto_controller/bayisian_possibility",
            Float32,
            self.bayesian_filter_callback,
        )
        self.bayesian_possibility = 0.0
        self.bayesian_threshold = 0.5

        # Distance
        self.distance_sub = rospy.Subscriber(
            "/intention/distance", Float32, self.distance_callback
        )
        self.eef_distance = float("inf")
        self.eef_distance_threshold = 0.2

    def distance_callback(self, msg: Float32):
        self.eef_distance = msg.data

    def right_controller_joy_callback(self, msg: Joy):
        # 오른쪽 - [A, B, 중지, 검지]
        if len(msg.buttons) != 4:
            # 예외 처리
            rospy.logwarn("Invalid Right Controller Joy Message")
            return None

        # A 버튼이 눌렸을 때, Vacuum Gripper 온 전환
        if msg.buttons[0] == 1:
            self.vacuum_gripper_state = True
            vacuum_command = "on" if self.vacuum_gripper_state else "off"
            self.vacuum_gripper_pub.publish(vacuum_command)

        # B 버튼이 눌렸을 때, Vacuum Gripper 오프 전환
        if msg.buttons[1] == 1:
            self.vacuum_gripper_state = False
            vacuum_command = "on" if self.vacuum_gripper_state else "off"
            self.vacuum_gripper_pub.publish(vacuum_command)

        # 중지 버튼이 눌렸을 때, 강제 수동 제어
        if msg.buttons[2] == 1:

            # 자동 상태에서 강제 수동 제어로 변경
            if self.control_mode == self.ControlMode.AUTO:
                rospy.loginfo("Change to Force Control")
                self.control_mode = self.ControlMode.FORCE_CONTROL

        # 중지 버튼이 눌리지 않았을 때, 강제 수동 제어 해제
        else:
            if self.control_mode == self.ControlMode.FORCE_CONTROL:
                self.control_mode = self.ControlMode.MANUAL

    def bayesian_filter_callback(self, msg: Float32):
        self.bayesian_possibility = msg.data

        if self.control_mode == self.ControlMode.MANUAL:
            # 수동 제어 상태에서 베이지안 확률이 높고, 거리가 작으면 자동으로 전환
            if (
                self.bayesian_possibility > self.bayesian_threshold
                and self.eef_distance < self.eef_distance_threshold
            ):
                rospy.loginfo("Bayesian Filter: Auto Control")
                self.control_mode = self.ControlMode.AUTO

                # Vacuum Gripper On
                self.vacuum_gripper_state = True
                vacuum_command = "on" if self.vacuum_gripper_state else "off"
                self.vacuum_gripper_pub.publish(vacuum_command)

    def control(self):
        manual_input = self.manual_controller.get_control_input()
        auto_input = self.auto_controller.get_control_input()

        combined_input = [0.0] * 6

        if self.control_mode == self.ControlMode.MANUAL:
            combined_input = manual_input

        elif self.control_mode == self.ControlMode.AUTO:
            combined_input = auto_input

        elif self.control_mode == self.ControlMode.FORCE_CONTROL:
            combined_input = manual_input

        self.rtde_c.speedL(combined_input, acceleration=0.25)


def main():
    rospy.init_node("control_test_node")  # TODO: Add node name

    hz = 20

    robot_control = URControl(hz=hz)

    homing = False

    if homing:
        # Home Position
        robot_control.rtde_c.moveJ(
            [
                -3.1415770689593714,
                -1.6416713200011195,
                -2.69926118850708,
                -1.9628821812071742,
                -1.570578400288717,
                3.141735315322876,
            ]
        )
        rospy.spin()

    else:
        r = rospy.Rate(hz)  # TODO: Add rate
        while not rospy.is_shutdown():

            robot_control.control()

            r.sleep()

    robot_control.rtde_c.stopL()


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
