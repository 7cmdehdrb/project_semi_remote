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
            return np.array([linear_vel.x, linear_vel.y, linear_vel.z, 0.0, 0.0, 0.0])

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

    def get_control_input(self) -> np.array:
        linear_vel = self.auto_controller_twist.linear
        return np.array([linear_vel.x, linear_vel.y, linear_vel.z, 0.0, 0.0, 0.0])


class URControl:
    class ControlMode(Enum):
        MANUAL = 0
        SEMI = 1
        FULL = 2

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
        self.control_mode = self.ControlMode.SEMI

        # Joy Stick
        self.right_controller_joy_sub = rospy.Subscriber(
            "/controller/right/joy", Joy, self.right_controller_joy_callback
        )

        # Bayisian Filter
        self.bayisian_filter_sub = rospy.Subscriber(
            "/auto_controller/bayisian_possibility",
            Float32,
            self.bayesian_filter_callback,
        )
        self.bayesian_possibility = 0.0
        self.bayesian_threshold = 0.5

    def right_controller_joy_callback(self, msg: Joy):
        # 오른쪽 - [A, B, 중지, 검지]
        if len(msg.buttons) != 4:
            rospy.logwarn("Invalid Right Controller Joy Message")
            return None

        if msg.buttons[2] == 1:
            # 중지 버튼이 눌렸을 때, 강제 수동 제어
            rospy.loginfo("Manual Control")
            self.control_mode = self.ControlMode.MANUAL

        else:
            # 중지 버튼이 눌리지 않았을 때, 반자율 제어
            self.control_mode = self.ControlMode.SEMI

    def bayesian_filter_callback(self, msg: Float32):
        self.bayesian_possibility = msg.data

        if (
            self.control_mode == self.ControlMode.SEMI
            and self.bayesian_possibility > self.bayesian_threshold
        ):
            rospy.loginfo("Bayesian Filter: Auto Control")
            self.control_mode = self.ControlMode.FULL

    def control(self):
        manual_input = self.manual_controller.get_control_input()
        auto_input = self.auto_controller.get_control_input()

        possibility = self.bayesian_possibility

        if self.control_mode == self.ControlMode.MANUAL:
            # 강제 수동 제어. 모든 입력을 수동 입력으로 변경
            possibility = 0.0
        elif self.control_mode == self.ControlMode.FULL:
            # 완전 제어. 모든 입력을 자동 입력으로 변경
            possibility = 1.0

        combined_input = possibility * auto_input + (1 - possibility) * manual_input

        self.rtde_c.speedL(combined_input, acceleration=0.25)


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
