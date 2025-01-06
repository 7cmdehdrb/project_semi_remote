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
    from vr_controller import URControl
except roslib.exceptions.ROSLibException:
    rospy.logfatal("Cannot find package test_package")
    sys.exit(1)


class BayisianFilter:
    def __init__(self):
        self.baysian_objects = {}
        self.small_value = 0.1

        self.hz = 10.0
        self.alpha = 1.0

        self.intersection_length_sub = rospy.Subscriber(
            "/intention/intersection_length", Int32, self.intersection_length_callback
        )
        self.intersection_length = 0

    def intersection_length_callback(self, msg: Int32):
        self.intersection_length = msg.data

    def parse_box_object_to_dict(self, normalized_boxes: BoxObjectMultiArrayWithPDF):
        # Max PDF에 대하여 정규화된 값. 이 값을 다시 정규화하여 사용해야 함.
        # 재정규화 안함. 병신같네 ㅅㅂ.
        baysian_objects = {}

        # 딕셔너리로 전환
        for box in normalized_boxes.boxes:
            box: BoxObjectWithPDF

            baysian_objects[str(box.id)] = box.pdf

            # clip도 intention 레벨에서 처리했음. 이제는 필요 없음.

        return baysian_objects

    def filter(
        self,
        boxes: BoxObjectMultiArrayWithPDF,
    ):

        # 새로 업데이트 되는 확률들
        objects = self.parse_box_object_to_dict(boxes)
        objects: dict

        # 초기화
        new_baysian_objects = {}

        # 새로운 오브젝트의 ID 리스트
        new_defined_list = list(set(objects.keys()) - set(self.baysian_objects.keys()))
        combined_list = list(set(objects.keys()) | set(self.baysian_objects.keys()))

        # 새로운 오브젝트들에 대한 초기 확률 설정 및 확률 업데이트. 초기확률은 균등확률로 지정
        for id in new_defined_list:
            input_possibility = 1.0 / len(combined_list)
            new_baysian_objects[str(id)] = {
                "possibility": float(input_possibility),
                "integral": 0.0,
            }

        dt = 1.0 / self.hz

        # 기존 오브젝트들에 대해 확률 업데이트, 적분 유지
        for id, value in self.baysian_objects.items():
            input_possibility = objects[str(id)]  # 우도로 취급

            possibility = float(value["possibility"])  # 이전 확률
            not_prior = 1.0 - possibility  # 이전 반대확률

            integral = float(value["integral"])  # 이전 적분

            # 우도
            likelihood = input_possibility  # 우도
            not_likelihood = 1.0 - input_possibility  # 반대 우도

            # 사후확률 계산
            post_probability = likelihood * possibility  # 사후확률
            post_not_probability = not_likelihood * not_prior  # 반대 사후확률

            # 정규화 상수
            normalized_constant = post_probability + post_not_probability

            # 정규화
            new_possibility = float(post_probability / normalized_constant)

            if self.intersection_length <= 5:
                new_possibility = 1.0 / len(combined_list)

            # 딕셔너리 업데이트
            new_baysian_objects[str(id)] = {
                "possibility": new_possibility,
                "integral": integral + (new_possibility * dt),
            }

        self.baysian_objects = new_baysian_objects

        return new_baysian_objects

    def reset(self):
        self.baysian_objects = {}

    def get_best_object(
        self,
    ):
        if len(self.baysian_objects.items()) == 0:
            # rospy.logwarn("The baysian objects are empty.")
            return -1, 0.0

        return -1, 0.0

        # 1. 확률이 0.9 이상인 경우 ID 반환
        best_object_id = max(
            self.baysian_objects, key=lambda x: self.baysian_objects[x]["possibility"]
        )
        best_object_possibility = float(
            self.baysian_objects[str(best_object_id)]["possibility"]
        )

        # 확률이 0.9 이상이고, 교차점이 1초 이상인 경우 반환
        if best_object_possibility >= 0.9 and self.intersection_length > int(
            self.hz * 1.0
        ):
            rospy.loginfo(f"최고 확률 임계: {best_object_possibility}")
            return int(best_object_id), True

        # 3초 이상 교차점이 있을 때, 확률이 0.5 이상인 경우 반환
        if best_object_possibility >= 0.5 and self.intersection_length > int(
            self.hz * 3.0
        ):
            rospy.loginfo(f"최고 확률 임계 및 교차점 만족: {best_object_possibility}")
            return int(best_object_id), True

        # possibility의 전체 합
        total_possibility = sum(
            [value["possibility"] for value in self.baysian_objects.values()]
        )

        normalized_possibility = best_object_possibility / total_possibility

        if (
            best_object_possibility >= 0.2
            and normalized_possibility >= 0.5
            and self.intersection_length > 20
        ):
            rospy.loginfo(
                f"최고 확률 임계 및 정규화 임계: {best_object_possibility}, {normalized_possibility}"
            )
            return int(best_object_id), True

        print(
            round(best_object_possibility, 5),
            round(normalized_possibility, 5),
            self.intersection_length,
        )

        # 4. 그 외의 경우, 실패
        return -1, False


class BoxManager:
    def __init__(self, topic: str = "/box_objects/pdf", callbacks: list = []):
        # Box Subscriber
        self.sub = rospy.Subscriber(
            topic, BoxObjectMultiArrayWithPDF, callback=self.callback
        )

        self.boxes_msg = BoxObjectMultiArrayWithPDF()
        self.callbacks = callbacks

    def callback(self, msg: BoxObjectMultiArrayWithPDF):
        # Max PDF 에 대하여 정규화 된 값이 들어옴. Intersection이 업데이트 될 때 반응함.
        self.boxes_msg = msg

        for callback in self.callbacks:
            callback: callable
            callback(msg)

    def get_pose(self, id: int):
        if id == -1:
            # 찾지 못한 경우
            return None

        if id == -2:
            # 홈 포지션
            return Pose(
                position=Point(
                    x=0.409971536841809, y=-0.13331905253904977, z=0.1163856447241224
                ),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            )

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

        self.control_mode = URControl.ControlMode.MANUAL
        self.best_object_id = -1
        self.is_success = False

        # EEF State
        self.eef_state = State(
            topic=None, frame_id="VGC"
        )  # VGC를 엔드 이펙터로 취급하는 Pose

        # Bayisian Filter
        self.bayisian_filter = BayisianFilter()

        # Box Manager
        self.box_manager = BoxManager(
            topic="/box_objects/pdf", callbacks=[self.bayisian_filter.filter]
        )

        # Control Mode Subscriber
        self.control_mode_sub = rospy.Subscriber(
            "/control_mode", UInt16, self.control_mode_callback
        )
        self.bayisian_success_pub = rospy.Publisher(
            "/auto_controller/bayisian_success", Bool, queue_size=1
        )

    def control_mode_callback(self, msg: UInt16):
        # Control Mode
        self.control_mode = URControl.ControlMode(msg.data)

        # 컨트롤 모드가 리셋일때, 베이지안 필터를 리셋함. 리셋을 콜벡에서 처리하는 이유는 동기화 때문임
        if self.control_mode == URControl.ControlMode.RESET:
            self.bayisian_filter.reset()

    # 로지스틱 함수 정의
    @staticmethod
    def logistic(x, midpoint, steepness, low, high):
        return low + (high - low) / (1 + np.exp(-steepness * (x - midpoint)))

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

    def straight_planning(
        self,
        target_pose: Pose,
        midpoint: float,
        steepness: float = 40,
    ) -> np.array:
        """midpoint 근처에서 속도가 빠르게 감소해서 x축으로 도달하면 속도가 0이 되도록 계산합니다."""
        current_eef_pose = self.eef_state.get_target_frame_pose()

        if current_eef_pose is None:
            rospy.logwarn("The current eef pose is None.")
            return np.array([0.0, 0.0, 0.0])

        # Position 만 넘파이 배열로 변환
        current_eef_pose_vector = np.array(
            [
                current_eef_pose.pose.position.x,
                current_eef_pose.pose.position.y,
                current_eef_pose.pose.position.z,
            ]
        )

        z_offset = 0.00  # Funcking Magic Number
        target_pose_vector = np.array(
            [
                target_pose.position.x,
                target_pose.position.y,
                target_pose.position.z + z_offset,
            ]
        )

        # 해당 포인트로 이동하기 위한 속도 벡터 생성, dt초에 이동
        velocity_vector = target_pose_vector - current_eef_pose_vector

        x_distance = velocity_vector[0]
        distance = np.abs(np.linalg.norm(velocity_vector))

        if np.abs(x_distance) <= 0.04:
            return np.array([0.0, 0.0, 0.0])

        velocity_vector = (
            velocity_vector / distance
        ) * ManipulatorPathPlanner.logistic(distance, midpoint, steepness, 0.05, 1.0)

        return velocity_vector

    def run(self):
        # 베이지안 필터는 BoxManager의 콜백 함수를 통해 자동으로 업데이트 됨.

        # @@@@@ 컨트롤 모드에 대한 예외 처리 @@@@@
        if self.control_mode == URControl.ControlMode.MANUAL:
            # 매뉴얼 모드일 때만, best_object_id를 업데이트함. 그 외는 업데이트 되지 않음
            self.best_object_id, self.is_success = (
                self.bayisian_filter.get_best_object()
            )

        elif self.control_mode == URControl.ControlMode.FORCE_CONTROL:
            # 힘 제어 모드일 때는 best_object_id를 -1로 설정함
            self.best_object_id = -1
            self.is_success = False

        elif self.control_mode == URControl.ControlMode.HOMING:
            # 홈 모드일 때는 best_object_id를 -2로 설정하고, 베이지안 필터를 초기화함
            # self.bayisian_filter.reset()
            self.best_object_id = -2
            self.is_success = False

        # get_pose() 함수는 -1에 대하여 None, -2에 대하여 홈 포지션을 반환함
        best_object_pose = self.box_manager.get_pose(self.best_object_id)

        if best_object_pose is None:
            # rospy.logwarn("Cannot Find Best Object: {}".format(self.best_object_id))
            return

        linear_velocity = self.straight_planning(
            target_pose=best_object_pose, midpoint=0.1, steepness=40
        )  # 1.0으로 정규화된 속도

        # Publish the linear velocity
        linear_velocity_msg = Twist()
        linear_velocity_msg.linear = Vector3(
            x=linear_velocity[0], y=linear_velocity[1], z=linear_velocity[2]
        )

        self.linear_velocity_pub.publish(linear_velocity_msg)
        self.bayisian_success_pub.publish(Bool(data=self.is_success))


def main():
    rospy.init_node("manipulator_path_planner_node")  # TODO: Add node name

    manipulator_path_planner = ManipulatorPathPlanner(move_group_name="manipulator")

    r = rospy.Rate(30)  # TODO: Add rate
    while not rospy.is_shutdown():

        manipulator_path_planner.run()

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
