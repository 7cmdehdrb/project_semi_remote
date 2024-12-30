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

# Moveit modules
from moveit_commander import RobotCommander, PlanningSceneInterface, MoveGroupCommander
from moveit_msgs.srv import (
    GetStateValidity,
    GetStateValidityRequest,
    GetStateValidityResponse,
)
from moveit_msgs.msg import RobotTrajectory, DisplayTrajectory, PlanningScene
from trajectory_msgs.msg import JointTrajectoryPoint

try:
    sys.path.append(roslib.packages.get_pkg_dir("base_package") + "/src")
    from state import State, EEFTwistState
except roslib.exceptions.ROSLibException:
    rospy.logfatal("Cannot find package test_package")
    sys.exit(1)


class BayisianFilter:
    def __init__(self):
        self.baysian_objects = {}

    def parse_box_object_to_dict(self, normalized_boxes: BoxObjectMultiArrayWithPDF):
        baysian_objects = {}

        for box in normalized_boxes.boxes:
            box: BoxObjectWithPDF

            baysian_objects[str(box.id)] = float(box.pdf)

        self.baysian_objects = baysian_objects

    def get_normalized_constant(self, objects: dict):
        if len(self.baysian_objects.items()) < len(objects.items()):
            """목표 객체의 개수가 변했을 때. 줄어드는 경우는 넘겨주는 레벨에서 잘못한 것임."""
            rospy.logwarn("The number of objects has changed.")

            reseted_baysian_objects = {
                {str(id): (1.0 / len(objects.items()))} for id in objects.keys()
            }
            self.baysian_objects = reseted_baysian_objects

        normanlized_constant = 0.0

        for id, possibility in self.baysian_objects.items():
            input_possibility = objects[str(id)]
            normanlized_constant += input_possibility * possibility

        return normanlized_constant

    def filter(self, boxes: BoxObjectMultiArrayWithPDF):
        objects = self.parse_box_object_to_dict(boxes)
        normalized_constant = self.get_normalized_constant(objects)

        if normalized_constant is None:
            return None

        new_baysian_objects = {}

        for id, possibility in self.baysian_objects.items():
            input_possibility = objects[str(id)]
            new_possibility = (input_possibility * possibility) / normalized_constant
            new_baysian_objects[str(id)] = float(new_possibility)

        self.baysian_objects = new_baysian_objects

        return new_baysian_objects

    def get_best_object(self):
        best_object_id = max(self.baysian_objects, key=self.baysian_objects.get)
        return int(best_object_id), float(self.baysian_objects[best_object_id])


class BoxManager:
    def __init__(self, topic: str = "/intention/real/pdf"):
        # Box Subscriber
        self.sub = rospy.Subscriber(
            topic, BoxObjectMultiArrayWithPDF, callback=self.callback
        )

        self.boxes_msg = BoxObjectMultiArrayWithPDF()
        self.normalized_boxes = BoxObjectMultiArrayWithPDF()

    def callback(self, msg: BoxObjectMultiArrayWithPDF):
        self.boxes_msg = msg
        self.normalized_boxes = self.normaize(msg)

    def normaize(self, msg: BoxObjectMultiArrayWithPDF):
        normalized_boxes = BoxObjectMultiArrayWithPDF()

        normalized_boxes.header = msg.header

        total_pdf = 0.0

        for box in msg.boxes:
            box: BoxObjectWithPDF
            total_pdf += box.pdf

        for box in msg.boxes:
            box: BoxObjectWithPDF

            normalized_box = BoxObjectWithPDF()

            normalized_box.header = box.header
            normalized_box.id = box.id
            normalized_box.pdf = box.pdf / total_pdf

            normalized_boxes.boxes.append(normalized_box)

        return normalized_boxes

    def get_pose(self, id: int):
        for box in self.boxes_msg.boxes:
            box: BoxObjectWithPDF

            if box.id == id:
                return box.pose

        rospy.logwarn("The object with id {} does not exist.".format(id))
        return None


class ManipulatorPathPlanner:
    def __init__(self, move_group_name: str = "ENDEFFECTOR"):
        self.bayisian_filter_pub = rospy.Publisher(
            "/auto_controller/bayisian_possibility", Float32, queue_size=1
        )
        self.linear_velocity_pub = rospy.Publisher(
            "/auto_controller/twist", Twist, queue_size=1
        )

        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group = MoveGroupCommander(move_group_name)

        """Moveit Parameter Group"""
        # self.group.set_planner_id("RRTConnectkConfigDefault")
        self.group.set_num_planning_attempts(10)
        # self.group.set_max_velocity_scaling_factor(0.1)
        # self.group.set_max_acceleration_scaling_factor(0.1)
        self.group.set_pose_reference_frame("map")

        self.trajectory = None

        # EEF State
        self.eef_state = State(
            topic=None, frame_id="VGC"
        )  # VGC를 엔드 이펙터로 취급하는 Pose

        # Box Manager
        self.box_manager = BoxManager(topic="/intention/real/pdf")

        # Bayisian Filter
        self.bayisian_filter = BayisianFilter()

        # Joint Converter
        self.joint_order = EEFTwistState.JointOrder()

    @staticmethod
    def dh_transform(a, d, alpha, theta):
        """
        Denavit-Hartenberg 변환 행렬 생성 함수.
        :param a: 링크 길이
        :param d: 링크 오프셋
        :param alpha: 링크 간 회전
        :param theta: 조인트 각도
        :return: 4x4 변환 행렬
        """
        return np.array(
            [
                [
                    np.cos(theta),
                    -np.sin(theta) * np.cos(alpha),
                    np.sin(theta) * np.sin(alpha),
                    a * np.cos(theta),
                ],
                [
                    np.sin(theta),
                    np.cos(theta) * np.cos(alpha),
                    -np.cos(theta) * np.sin(alpha),
                    a * np.sin(theta),
                ],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0, 0, 0, 1],
            ]
        )

    @staticmethod
    def forward_kinematics(joint_angles):
        """
        UR5e Forward Kinematics 계산 함수.
        :param joint_angles: 길이 6짜리 NumPy 배열, 조인트 각도 (라디안)
        :return: End Effector Pose (4x4 변환 행렬)
        """

        # UR5e DH 파라미터
        dh_params = [
            # (a_i, d_i, alpha_i, theta_i)
            (0, 0.1625, np.pi / 2, joint_angles[0]),  # Joint 1
            (-0.425, 0, 0, joint_angles[1]),  # Joint 2
            (-0.3922, 0, 0, joint_angles[2]),  # Joint 3
            (0, 0.1333, np.pi / 2, joint_angles[3]),  # Joint 4
            (0, 0.0997, -np.pi / 2, joint_angles[4]),  # Joint 5
            (0, 0.0996, 0, joint_angles[5]),  # Joint 6
        ]

        # 초기 변환 행렬 (Identity Matrix)
        t_matrix = np.eye(4)

        # X, Y축 뒤집기 행렬. 왜 이런지는 모르겠음.
        flip_transform = np.array(
            [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        # 각 DH 파라미터를 적용하여 누적 변환 계산
        for a, d, alpha, theta in dh_params:
            t_matrix = np.dot(
                t_matrix, ManipulatorPathPlanner.dh_transform(a, d, alpha, theta)
            )

        return np.dot(flip_transform, t_matrix)

    def convert_joint_trajectory_to_eef_trajectory(
        self, joint_trajectory: RobotTrajectory
    ) -> np.array:
        """
        shoulder_pan_joint,
        shoulder_lift_joint,
        elbow_joint,
        wrist_1_joint,
        wrist_2_joint,
        wrist_3_joint,
        해당 순서로 메세지를 변환합니다.
        """

        do_passing = False

        if self.joint_order.order == joint_trajectory.joint_trajectory.joint_names:
            do_passing = True

        poses = np.empty((0, 3))

        for point in joint_trajectory.joint_trajectory.points:
            point: JointTrajectoryPoint

            if do_passing:
                # 주어진 순서와 같으면, 바로 넘파이 배열로 변환
                joint_position = np.array(point.positions)

            else:
                joint_states = JointState()

                joint_states.name = joint_trajectory.joint_trajectory.joint_names
                joint_states.position = point.positions

                sorted_joint_states = self.joint_order.change_idx(joint_states)
                joint_position = np.array(sorted_joint_states.position)

            joint_position = np.array(joint_position)
            eef_pose = ManipulatorPathPlanner.forward_kinematics(joint_position)[
                :3, 3
            ]  # End Effector Pose, 길이 3 벡터

            # 배열에 추가
            poses = np.vstack((poses, eef_pose))

        return poses

    def planning(self, target_pose: Pose, dt: float = 0.1) -> np.array:
        self.group.set_pose_target(target_pose)

        is_success, trajectory, _, _ = self.group.plan()
        trajectory: RobotTrajectory

        # EEF Update
        # Map Frame 기준 EEF Pose
        current_eef_pose = self.eef_state.get_target_frame_pose()

        # Position 만 넘파이 배열로 변환
        current_eef_pose_vector = np.array(
            [
                current_eef_pose.pose.position.x,
                current_eef_pose.pose.position.y,
                current_eef_pose.pose.position.z,
            ]
        )

        # 새로운 경로 탐색에 성공한 경우
        if is_success:

            # 경로 저장
            self.trajectory = trajectory

            # 가장 짧은 경로점을 찾음
            nearest_point = trajectory.joint_trajectory.points[0]
            nearest_point: JointTrajectoryPoint

            joint_positions = np.array(nearest_point.positions)

            # EEF Pose 계산
            best_point = ManipulatorPathPlanner.forward_kinematics(joint_positions)[
                :3, 3
            ]

            # 해당 포인트로 이동하기 위한 속도 벡터 생성, dt초에 이동
            velocity_vector = (best_point - current_eef_pose_vector) / dt

            # 선속도 벡터 리턴
            return velocity_vector

        elif not is_success and self.trajectory is not None:
            # 이전 경로를 사용하여 최단 경로점을 찾음.

            # Joint Trajectory를 EEF Trajectory 넘파이 배열로 변환
            eef_poses = self.convert_joint_trajectory_to_eef_trajectory(self.trajectory)

            # 현재 EEF 위치와 목표 EEF 위치 사이의 거리 계산
            distances = np.linalg.norm(eef_poses - current_eef_pose_vector, axis=1)

            # 거리가 가장 짧은 포인트 선택
            best_point_idx = np.argmin(distances)
            best_point = eef_poses[best_point_idx]

            # 해당 포인트로 이동하기 위한 속도 벡터 생성, dt초에 이동
            velocity_vector = (best_point - current_eef_pose_vector) / dt

            # 선속도 벡터 리턴
            return velocity_vector

        # 경로 탐색에 실패한 경우, 예외 사항, 정지 명령
        rospy.logwarn("Planning failed.")
        return np.array([0.0, 0.0, 0.0])

    def run(self):
        _ = self.bayisian_filter.filter(
            boxes=self.box_manager.normalized_boxes
        )  # Automatically update the baysian filter

        best_object_id, best_object_possibility = (
            self.bayisian_filter.get_best_object()
        )  # Get the best object, Dictionary type

        best_object_pose = self.box_manager.get_pose(best_object_id)

        if best_object_pose is None:
            rospy.logwarn("The best object pose is None.")
            return

        linear_velocity = self.planning(target_pose=best_object_pose)

        # Publish the linear velocity
        linear_velocity_msg = Twist()
        linear_velocity_msg.linear = Vector3(
            x=linear_velocity[0], y=linear_velocity[1], z=linear_velocity[2]
        )

        # Publish the baysian possibility
        baysian_possibility_msg = Float32()
        baysian_possibility_msg.data = best_object_possibility

        self.linear_velocity_pub.publish(linear_velocity_msg)
        self.bayisian_filter_pub.publish(baysian_possibility_msg)


def main():
    rospy.init_node("manipulator_path_planner_node")  # TODO: Add node name

    manipulator_path_planner = ManipulatorPathPlanner()

    r = rospy.Rate(10)  # TODO: Add rate
    while not rospy.is_shutdown():

        manipulator_path_planner.run()

        r.sleep()


if __name__ == "__main__":
    try:
        main()
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
