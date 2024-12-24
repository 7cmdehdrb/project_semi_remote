#!/usr/bin/python3

import os
import sys
import roslib.exceptions
import roslib.packages
import rospy
import roslib
import numpy as np
from abc import ABC, abstractmethod

# Moveit!
import moveit_commander

# Messages
from tf.transformations import (
    quaternion_multiply,
    quaternion_from_euler,
    euler_from_quaternion,
)
from std_msgs.msg import *
from sensor_msgs.msg import *
from geometry_msgs.msg import *
from nav_msgs.msg import *
from custom_msgs.msg import *

# Custom
try:
    sys.path.append(roslib.packages.get_pkg_dir("test_package") + "/src")
    from abstract_state import State as AbstractState
except roslib.exceptions.ROSLibException:
    rospy.logfatal("Cannot find package test_package")
    sys.exit(1)


class EndEffectorState(AbstractState):
    def __init__(self, topic):
        super().__init__(topic)

        # ROS
        self.pub = rospy.Publisher(topic, Odometry, queue_size=10)
        self.odom = Odometry()

        # Moveit
        self.move_group = moveit_commander.MoveGroupCommander(
            "ENDEFFECTOR"
        )  # 플래닝 그룹 이름

        # variables
        self.rotate_quat = quaternion_from_euler(0.0, -np.pi / 2.0, 0.0)

    def calculate_twist(self, odom1: Odometry, odom2: Odometry):
        dt = (odom2.header.stamp - odom1.header.stamp).to_sec()
        if dt == 0.0:
            rospy.logerr("Time difference (dt) cannot be zero.")
            return Vector3(0, 0, 0), Vector3(0, 0, 0)

        # 위치 변화 계산
        dx = odom2.pose.pose.position.x - odom1.pose.pose.position.x
        dy = odom2.pose.pose.position.y - odom1.pose.pose.position.y

        # 쿼터니언 -> 오일러 각 변환
        def extract_euler(odom):
            q = odom.pose.pose.orientation
            return euler_from_quaternion([q.x, q.y, q.z, q.w])

        roll1, pitch1, yaw1 = extract_euler(odom1)
        roll2, pitch2, yaw2 = extract_euler(odom2)

        # 각도 변화 계산
        droll, dpitch, dyaw = roll2 - roll1, pitch2 - pitch1, yaw2 - yaw1

        # 선속도와 각속도 계산
        linear_velocity = Vector3(dx / dt, dy / dt, 0.0)
        angular_velocity = Vector3(droll / dt, dpitch / dt, dyaw / dt)

        return linear_velocity, angular_velocity

    def get_odom(self):
        odom = Odometry()

        ee_pose = self.move_group.get_current_pose().pose

        odom.header = Header(frame_id="base_link", stamp=rospy.Time.now())

        quat = [
            ee_pose.orientation.x,
            ee_pose.orientation.y,
            ee_pose.orientation.z,
            ee_pose.orientation.w,
        ]  # original quaternion
        rquat = quaternion_multiply(quat, self.rotate_quat)  # result quaternion

        odom.pose.pose.position = ee_pose.position
        odom.pose.pose.orientation = Quaternion(
            x=rquat[0], y=rquat[1], z=rquat[2], w=rquat[3]
        )

        linear, angular = self.calculate_twist(self.odom, odom)

        odom.twist.twist.linear = linear
        odom.twist.twist.angular = angular

        return odom

    def publish_odom(self):
        self.odom = self.get_odom()
        self.pub.publish(self.odom)

    def odom_callback(self, msg: Odometry):
        """This callback function input nav_msgs.msg.Odometry and update robot's state"""
        if not isinstance(msg, Odometry):
            rospy.logerr("Received message is not of type Odometry")
            return

        self.notify_observers()

    def register_observer(self, observer_callback):
        self._observers.append(observer_callback)

    def notify_observers(self):
        """Call all observers function when this function is called"""
        for observer in self._observers:
            observer()


def main():
    rospy.init_node("end_effector_state_node", anonymous=True)

    end_effector_state = EndEffectorState("/odometry/end_effector")

    r = rospy.Rate(10)
    while not rospy.is_shutdown():

        end_effector_state.publish_odom()

        r.sleep()


if __name__ == "__main__":
    main()
