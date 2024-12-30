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
from custom_msgs.msg import *

# Custom
from abc import ABC

# MoveIt
from moveit_commander import RobotCommander, MoveGroupCommander


class Localization(ABC):
    def __init__(self, topic: str, frame_id: str):
        super().__init__()

        # Basic information
        self.topic = topic
        self.frame_id = frame_id

        # TF
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()


class EEFTwistState:
    class JointOrder:
        def __init__(self):
            self.order = [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ]

        def set_order(self, order: list):
            self.order = order

        def change_idx(self, joint_state: JointState):
            new_joint_state = JointState()
            new_joint_state.name = self.order

            if not all(joint_name in joint_state.name for joint_name in self.order):
                raise ValueError("Invalid joint names")

            if len(joint_state.position) != len(joint_state.name):
                new_joint_state.position = [0.0] * len(self.order)
            else:
                new_joint_state.position = [
                    joint_state.position[joint_state.name.index(joint_name)]
                    for joint_name in self.order
                ]

            if len(joint_state.velocity) != len(joint_state.name):
                new_joint_state.velocity = [0.0] * len(self.order)
            else:
                new_joint_state.velocity = [
                    joint_state.velocity[joint_state.name.index(joint_name)]
                    for joint_name in self.order
                ]

            if len(joint_state.effort) != len(joint_state.name):
                new_joint_state.effort = [0.0] * len(self.order)
            else:
                new_joint_state.effort = [
                    joint_state.effort[joint_state.name.index(joint_name)]
                    for joint_name in self.order
                ]

            return new_joint_state

    def __init__(self, movegroup_name: str = "ENDEFFECTOR"):
        self.orderer = self.JointOrder()
        self.move_commander = MoveGroupCommander(movegroup_name)

        self.joint_states_subscriber = rospy.Subscriber(
            "/joint_states", JointState, self.joint_states_callback
        )

        # MoveIt Commander

        self.orderer.set_order(self.move_commander.get_active_joints())

        self.joint_states = JointState(
            name=self.orderer.order,
            position=[0.0] * len(self.orderer.order),
            velocity=[0.0] * len(self.orderer.order),
            effort=[0.0] * len(self.orderer.order),
        )
        self.twist = Twist()

    def joint_states_callback(self, msg: JointState):
        """
        name:
        - shoulder_pan_joint
        - shoulder_lift_joint
        - elbow_joint
        - wrist_1_joint
        - wrist_2_joint
        - wrist_3_joint
        position: [0.6072, -1.326, 2.2864, 2.1192, 0.0, 0.0]
        velocity: []
        effort: []
        """
        self.joint_states = self.orderer.change_idx(msg)
        self.twist = self.calculate_linear_velocities(self.joint_states)

    def calculate_linear_velocities(self, joint_states: JointState):
        J = self.move_commander.get_jacobian_matrix(
            self.move_commander.get_current_joint_values()
        )

        joint_velocities = np.array(self.joint_states.velocity)
        linear_velocity = np.dot(J, joint_velocities)

        twist = Twist()

        twist.linear = Vector3(
            x=linear_velocity[0], y=linear_velocity[1], z=linear_velocity[2]
        )
        twist.angular = Vector3(
            x=linear_velocity[3], y=linear_velocity[4], z=linear_velocity[5]
        )

        return twist


class State(Localization):
    def __init__(self, topic: str, frame_id: str):
        super().__init__(topic, frame_id)

        self.data = None
        self.pose = None

        if self.topic is not None:
            self.subscriber = rospy.Subscriber(
                self.topic, Odometry, self.odom_callback, queue_size=1
            )

    def odom_callback(self, msg: Odometry):
        if not isinstance(msg, Odometry):
            rospy.logerr("Invalid message type")

        self.data = msg
        self.get_target_frame_pose()

    def get_target_frame_pose(self):
        """Get position and orientation of specified frame_id in map frame"""
        if self.tf_listener.canTransform("map", self.frame_id, time=rospy.Time()):
            origin_pose = PoseStamped()
            origin_pose.header = Header(frame_id=self.frame_id, stamp=rospy.Time())
            origin_pose.pose.orientation = Quaternion(
                x=0.0, y=0.0, z=0.0, w=1.0
            )  # Base Pose

            transfromed_pose = self.tf_listener.transformPose("map", origin_pose)

            self.pose = transfromed_pose

            return self.pose
        else:
            rospy.logerr(f"Cannot get transform from map to {self.frame_id}")
            return None

    def get_target_frame_pose_on_target_frame(self, target_frame: str):
        if self.tf_listener.canTransform(
            target_frame, self.frame_id, time=rospy.Time()
        ):
            origin_pose = PoseStamped()
            origin_pose.header = Header(frame_id=self.frame_id, stamp=rospy.Time())
            origin_pose.pose.orientation = Quaternion(
                x=0.0, y=0.0, z=0.0, w=1.0
            )  # Base Pose

            transfromed_pose = self.tf_listener.transformPose(target_frame, origin_pose)

            self.pose = transfromed_pose

            return self.pose
        else:
            rospy.logerr(f"Cannot get transform from {target_frame} to {self.frame_id}")
            return None


if __name__ == "__main__":
    rospy.init_node("test_node")

    rospy.spin()
