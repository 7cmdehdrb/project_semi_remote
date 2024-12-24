#!/usr/bin/python3

import os
import sys
import roslib.exceptions
import roslib.packages
import rospy
import roslib
import numpy as np
import cv2
from cv_bridge import CvBridge
from abc import ABC, abstractmethod

from tf.transformations import euler_from_quaternion, quaternion_from_euler
import moveit_commander

# Messages
from std_msgs.msg import *
from sensor_msgs.msg import *
from geometry_msgs.msg import *
from nav_msgs.msg import *
from custom_msgs.msg import *


class BoxesState:
    def __init__(self):
        self.data = BoxObjectMultiArray()

        self.boxes_sub = rospy.Subscriber(
            "/segmetation/boxes", BoxObjectMultiArray, self.boxes_callback
        )

    def boxes_callback(self, msg):
        if not isinstance(msg, BoxObjectMultiArray):
            rospy.logerr("Received message is not of type BoxObjectMultiArray")
            return

        rospy.loginfo("Successfully subscribe BoxObjectMultiArray")

        self.data = msg

        self.boxes_sub.unregister()


class EndEffectorControl:
    def __init__(self):

        self.boxes = BoxesState()

        # Moveit
        self.move_group = moveit_commander.MoveGroupCommander(
            "ENDEFFECTOR"
        )  # 플래닝 그룹 이름
        # MoveIt! 초기화
        moveit_commander.roscpp_initialize(sys.argv)

        # Move Group Commander 초기화 (플래닝 그룹 설정)
        self.move_group = moveit_commander.MoveGroupCommander("ENDEFFECTOR")

        self.move_group.set_max_velocity_scaling_factor(0.3)  # 100% 속도
        self.move_group.set_max_acceleration_scaling_factor(0.3)  # 100% 가속도

    def calculate_traj(self, idx: int):
        if len(self.boxes.data.boxes) <= idx:
            rospy.logerr("Index out of range")
            return None

        if idx == -1:
            target_box = BoxObject(
                id=0,
                pose=Pose(
                    position=Point(x=0.0, y=-0.5, z=0.5),
                    orientation=Quaternion(x=0, y=0, z=0, w=1),
                ),
                scale=Vector3(x=0.1, y=0.1, z=0.1),
            )
        else:
            target_box = self.boxes.data.boxes[idx]

        # calculate distanse between end effector and target box
        current_pose = self.move_group.get_current_pose().pose
        target_pose = target_box.pose

        distance = np.sqrt(
            (current_pose.position.x - target_pose.position.x) ** 2
            + (current_pose.position.y - target_pose.position.y) ** 2
            + (current_pose.position.z - target_pose.position.z) ** 2
        )

        if distance < 0.1:
            # rospy.loginfo("Distance is less than 0.1")
            return None

        # 타겟 초기화
        self.move_group.clear_pose_targets()

        target_box.pose.position.x -= 0.125  # 12.5cm 뒤로 이동

        quat = quaternion_from_euler(0.0, np.pi / 2.0, 0.0)
        p = Pose(
            position=target_box.pose.position,
            orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
        )

        self.move_group.set_pose_target(p)

        plan_success, robot_trajectory, planning_time, _ = (
            self.move_group.plan()
        )  # 계획 생성

        if plan_success:
            # rospy.loginfo("Plan was found: %f", planning_time)
            return robot_trajectory
        else:
            rospy.logerr("Plan was not found.")
            # return None

    def execute_traj(self, traj):
        if traj is None:
            # rospy.logerr("Trajectory is None")
            return

        self.move_group.execute(traj, wait=True)
        self.move_group.stop()  # 로봇 정지

        # 목표 Pose 설정 해제
        self.move_group.clear_pose_targets()


def main():
    # ROS 노드 초기화
    rospy.init_node("move_robot_to_pose")

    end_effector_control = EndEffectorControl()

    # rospy.spin()

    r = rospy.Rate(1)
    while not rospy.is_shutdown():
        idx = rospy.get_param(
            "/move_robot_to_pose/idx", 0
        )  # Default value is 0 if param is not set

        traj = end_effector_control.calculate_traj(idx)
        end_effector_control.execute_traj(traj)
        r.sleep()


if __name__ == "__main__":
    main()
