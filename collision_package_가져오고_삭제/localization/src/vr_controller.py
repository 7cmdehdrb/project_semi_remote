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
import time


class URControl:
    class ControlMode(Enum):
        STOP = 0
        LINEAR = 1
        ANGULAR = 2

    def __init__(self, hz: int = 20):
        self.hz = hz

        IP = "192.168.2.2"

        # RTDE Control and Receive Interface
        self.rtde_c = rtde_control.RTDEControlInterface(IP)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(IP)

        # Right Controller Data : Initialize
        self.right_controller_twist = Twist()
        self.right_controller_joy = Joy()
        self.right_controller_joy.buttons = [0, 0, 0, 0]

        # Left Controller Data : Initialize
        self.left_controller_joy = Joy()
        self.left_controller_joy.buttons = [0, 0, 0, 0]

        # Spine Pose Data : Initialize
        self.spine_pose = Pose()
        self.angular_z_speed = 0.0
        self.last_time = rospy.Time.now()

        self.control_mode = self.ControlMode.STOP

        self.collision_is_valid = True

        # Controller, Spine, Collision Subscriber
        self.right_controller_twist_sub = rospy.Subscriber(
            "/controller/right/twist", Twist, self.right_controller_callback
        )
        self.right_controller_joy_sub = rospy.Subscriber(
            "/controller/right/joy", Joy, self.right_controller_joy_callback
        )
        self.left_controller_joy_sub = rospy.Subscriber(
            "/controller/left/joy", Joy, self.left_controller_joy_callback
        )
        self.spine_pose_sub = rospy.Subscriber(
            "/spine/pose", Pose, self.spine_pose_callback
        )
        self.collision_sub = rospy.Subscriber(
            "/is_collision", Bool, self.collision_callback
        )

    def collision_callback(self, msg: Bool):
        self.collision_is_valid = msg.data

    def right_controller_callback(self, msg: Twist):
        self.right_controller_twist = msg

    def right_controller_joy_callback(self, msg: Joy):
        self.right_controller_joy = msg
        # 오른쪽 - [A, B, 중지, 검지]
        # 왼쪽 - [X, Y, 중지, 검지]

    def left_controller_joy_callback(self, msg: Joy):
        self.left_controller_joy = msg

    def spine_pose_callback(self, msg: Pose):
        current_time = rospy.Time.now()

        dt = (current_time - self.last_time).to_sec()

        # Calculate angular z speed
        if dt > 0:
            last_quaternion = [
                self.spine_pose.orientation.x,
                self.spine_pose.orientation.y,
                self.spine_pose.orientation.z,
                self.spine_pose.orientation.w,
            ]

            current_quaternion = [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ]

            delta_quaternion = quaternion_multiply(
                quaternion_inverse(last_quaternion), current_quaternion
            )

            _, _, delta_yaw = euler_from_quaternion(delta_quaternion)

            self.angular_z_speed = delta_yaw / dt
        else:
            rospy.logwarn("dt is less than 0.")

        self.last_time = current_time
        self.spine_pose = msg

    def linear_control(self):
        linear_speed = self.right_controller_twist.linear

        linear_speed.x = np.clip(linear_speed.x, -0.5, 0.5)
        linear_speed.y = np.clip(linear_speed.y, -0.5, 0.5)
        linear_speed.z = np.clip(linear_speed.z, -0.5, 0.5)

        input_data = [linear_speed.y, -linear_speed.x, linear_speed.z, 0, 0, 0]

        return input_data

    def angular_control(self):
        input_data = [np.clip(-self.angular_z_speed, -0.5, 0.5), 0, 0, 0, 0, 0]

        return input_data

    def control(self):
        # 충돌이 발생했을 때, 로봇을 정지시킴
        if self.collision_is_valid is False:
            self.rtde_c.speedL([0, 0, 0, 0, 0, 0], acceleration=0.25)
            return None

        # 오른손 검지 버튼이 눌렸을 때, 모든 종류의 동작 명령이 기동
        if self.right_controller_joy.buttons[3] == 1:

            # 왼손 검지 버튼이 눌렸을 때, 회전 명령
            if self.left_controller_joy.buttons[3] == 1:

                # 만약 직선 명령이 진행 중이었다면, 정지 명령을 내린 후 회전 명령을 내림
                if self.control_mode != self.ControlMode.ANGULAR:
                    self.rtde_c.speedL([0, 0, 0, 0, 0, 0], acceleration=0.25)
                    # time.sleep(0.5)

                # 만약 회전 명령이 진행 중이었다면, 회전 명령을 계속 진행
                else:
                    angular_control_msg = self.angular_control()
                    self.rtde_c.speedJ(
                        angular_control_msg, acceleration=0.25, time=(1 / self.hz)
                    )

                # 회전 명령으로 전환
                self.control_mode = self.ControlMode.ANGULAR

            # 왼손 검지 버튼이 눌리지 않았을 때, 직선 명령
            else:

                # 만약 회전 명령이 진행 중이었다면, 정지 명령을 내린 후 직선 명령을 내림
                if self.control_mode != self.ControlMode.LINEAR:
                    self.rtde_c.speedJ([0, 0, 0, 0, 0, 0], acceleration=0.25)
                    # time.sleep(0.5)

                # 만약 직선 명령이 진행 중이었다면, 직선 명령을 계속 진행
                else:
                    linear_control_msg = self.linear_control()
                    self.rtde_c.speedL(
                        linear_control_msg, acceleration=0.25, time=(1 / self.hz)
                    )

                # 직선 명령으로 전환
                self.control_mode = self.ControlMode.LINEAR

        # 오른손 검지 버튼이 눌리지 않았을 때, 정지 명령
        else:
            self.rtde_c.speedL([0, 0, 0, 0, 0, 0], acceleration=0.25)


def main():
    rospy.init_node("control_test_node")  # TODO: Add node name

    hz = 20

    robot_control = URControl(hz=hz)

    r = rospy.Rate(hz)  # TODO: Add rate
    while not rospy.is_shutdown():

        robot_control.control()

        r.sleep()


if __name__ == "__main__":
    main()
    try:
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
