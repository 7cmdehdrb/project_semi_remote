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

# Custom
from moveit_commander import RobotCommander, PlanningSceneInterface, MoveGroupCommander


def normalize_euler(angles: list):
    result = []
    for angle in angles:
        for _ in range(10):
            if angle < np.pi:
                break
            angle -= np.pi * 2.0

        for _ in range(10):
            if angle > -np.pi:
                break
            angle += np.pi * 2.0

        result.append(angle)
    return result


class InverseKinematics(object):
    def __init__(self):
        self.joint_state = JointState()
        self.joint_state.header.stamp = rospy.Time.now()
        self.joint_state.name = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        # self.joint_state.position = [0.6072, -1.326, 2.2864, 2.1192, 0.0, 0.0]
        self.joint_state.position = [0.0, 0.0, 0.0, 0.0, np.pi / 2.0, 0.0]

        self.endeffector_twist = Twist()
        self.spine_pose = None
        self.joy_state = Joy(buttons=[0, 0, 0, 0])

        self.joint_state_publisher = rospy.Publisher(
            "/joint_states", JointState, queue_size=10
        )

        self.joystick_subscriber = rospy.Subscriber(
            "/left_controller/joy", Joy, self.joystick_callback
        )

        self.endeffector_twist_subscriber = rospy.Subscriber(
            "/left_controller/twist", Twist, self.endeffector_twist_callback
        )

        self.spine_pose_subscriber = rospy.Subscriber(
            "/spine/pose", PoseStamped, self.spine_pose_callback
        )

        self.robot_commader = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.move_group = MoveGroupCommander("ENDEFFECTOR")

    def joystick_callback(self, msg: Joy):
        self.joy_state = msg

    def spine_pose_callback(self, msg: PoseStamped):
        if self.spine_pose is None:
            self.spine_pose = msg
            return 0

        dt = (msg.header.stamp - self.spine_pose.header.stamp).to_sec()

        if dt < float(1.0 / 30.0):
            return 0

        last_quat = [
            self.spine_pose.pose.orientation.x,
            self.spine_pose.pose.orientation.y,
            self.spine_pose.pose.orientation.z,
            self.spine_pose.pose.orientation.w,
        ]

        current_quat = [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ]

        dquat = quaternion_multiply(quaternion_inverse(last_quat), current_quat)

        _, _, dtheta = euler_from_quaternion(dquat)

        # dz =
        # dz = np.clip(dz, -0.1, 0.1)

        angular_velocity = Vector3(x=0.0, y=0.0, z=dtheta / dt)

        self.endeffector_twist.angular = angular_velocity

        self.spine_pose = msg

    def endeffector_twist_callback(self, msg: Twist):
        x = msg.linear.x
        y = msg.linear.y
        z = msg.linear.z

        x = np.clip(x, -0.1, 0.1)
        y = np.clip(y, -0.1, 0.1)
        z = np.clip(z, -0.1, 0.1)

        self.endeffector_twist.linear = Vector3(x=x, y=y, z=z)

    def compute_joint_velocities(self, target_velocity: list):
        # 현재 조인트 상태 가져오기
        joint_values = self.joint_state.position

        # 자코비안 계산
        jacobian = self.move_group.get_jacobian_matrix(joint_values)

        # 엔드 이펙터의 목표 선속도를 설정 (예: [vx, vy, vz])
        target_velocity = np.array(target_velocity)  # [vx, vy, vz]

        # 자코비안의 유사역행렬 계산
        jacobian_pseudo_inverse = np.linalg.pinv(jacobian)

        # 조인트 속도 계산
        joint_velocities = np.dot(jacobian_pseudo_inverse, target_velocity)

        return joint_velocities

    def get_joint_state_message(self, joint_velocity: np.array):
        msg = JointState()
        msg.header = Header(stamp=rospy.Time.now())
        msg.name = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        dt = (msg.header.stamp - self.joint_state.header.stamp).to_sec()
        current_joint_values = np.array(self.joint_state.position)

        # 조인트 위치 계산
        new_joint_values = current_joint_values + joint_velocity * dt

        msg.position = normalize_euler(list(new_joint_values))

        self.joint_state = msg

        return msg

    def publish_joint_state(self):
        self.joint_state.header.stamp = rospy.Time.now()
        self.joint_state_publisher.publish(self.joint_state)

    def run(self):
        if self.joy_state.buttons[3] == 0:
            x = self.endeffector_twist.linear.x
            y = self.endeffector_twist.linear.y
            z = self.endeffector_twist.linear.z

            # 제한된 x, y, z 값 설정
            x = np.clip(x, -0.3, 0.3)
            y = np.clip(y, -0.3, 0.3)
            z = np.clip(z, -0.3, 0.3)

            linear_velocity = [
                x,
                y,
                z,
                0.0,
                0.0,
                0.0,
            ]

            joint_vel = self.compute_joint_velocities(linear_velocity)
            self.get_joint_state_message(joint_vel)

        elif self.joy_state.buttons[3] == 1:
            rospy.loginfo("Endeffector twist mode.")
            print(self.endeffector_twist.angular.z)
            joint_vel = np.array(
                [self.endeffector_twist.angular.z, 0.0, 0.0, 0.0, 0.0, 0.0]
            )
            self.get_joint_state_message(joint_vel)


def main():
    rospy.init_node("moveit_test_node")  # TODO: Add node name

    ik = InverseKinematics()

    r = rospy.Rate(10)  # TODO: Add rate

    for _ in range(10):
        ik.publish_joint_state()
        r.sleep()

    while not rospy.is_shutdown():
        ik.run()
        ik.publish_joint_state()
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
