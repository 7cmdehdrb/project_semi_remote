#!/usr/bin/python3

# Basic modules
import os
import sys
import roslib.exceptions
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
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.integrate import quad
from enum import Enum
import random

try:
    sys.path.append(roslib.packages.get_pkg_dir("base_package") + "/src")
    from state import State, EEFTwistState
    from vr_controller import URControl
except roslib.exceptions.ROSLibException:
    rospy.logfatal("Cannot find package test_package")
    sys.exit(1)

PoseWithCovarianceStamped

class Plane:
    def __init__(self, n: np.array, d: float):
        if not isinstance(n, np.ndarray):
            raise ValueError("n must be a numpy array")

        self.n = n
        self.d = d

    def distance(self, p: np.array):
        """Calculate the distance of a point to the plane"""
        return np.abs(np.dot(self.n, p) + self.d) / np.linalg.norm(self.n)

    def project_to_plane(self, p: np.array):
        """Project a 3D point onto the plane and return its 2D coordinates"""
        # Find a point on the plane
        point_on_plane = p - (np.dot(self.n, p) + self.d) * self.n

        # Create a local coordinate system on the plane
        u = np.array([-self.n[1], self.n[0], 0])  # Arbitrary vector orthogonal to n
        if np.allclose(u, 0):
            u = np.array([0, -self.n[2], self.n[1]])  # Handle edge case
        u = u / np.linalg.norm(u)  # Normalize
        v = np.cross(self.n, u)  # Ensure v is orthogonal to both n and u

        # Project the point onto the local 2D basis
        x = np.dot(point_on_plane, u)
        y = np.dot(point_on_plane, v)
        return np.array([x, y]), u, v

    def transform_covariance(self, cov_3d: np.array, u: np.array, v: np.array):
        """Transform a 3D covariance matrix to 2D on the plane"""
        # Transformation matrix from 3D to 2D
        T = np.stack((u, v), axis=0)  # Shape: (2, 3)
        # Project the covariance matrix
        cov_2d = T @ cov_3d @ T.T  # Shape: (2, 2)
        return cov_2d

    def transform_to_2d(self, position: np.array, covariance: np.array):
        """Transform a 3D position and covariance to 2D on the plane"""
        if not isinstance(position, np.ndarray) or not isinstance(
            covariance, np.ndarray
        ):
            raise ValueError("Position and covariance must be numpy arrays")
        if position.shape != (3,) or covariance.shape != (3, 3):
            raise ValueError(
                "Position must be a 3-element vector and covariance must be 3x3"
            )

        # Project position to 2D
        pos_2d, u, v = self.project_to_plane(position)

        # Transform covariance to 2D
        cov_2d = self.transform_covariance(covariance, u, v)

        return pos_2d, cov_2d


class BoxManager:
    """박스 객체를 관리하는 클래스. 박스 정보를 수집"""

    def __init__(self, topic: str = "/box_objects/raw"):
        self.boxes_subscriber = rospy.Subscriber(
            topic, BoxObjectMultiArrayWithPDF, self.boxes_callback
        )
        self.boxes_msg = BoxObjectMultiArrayWithPDF()

    def boxes_callback(self, msg: BoxObjectMultiArrayWithPDF):
        self.boxes_msg = msg


class RealIntentionGaussian:
    class IntersectionMethod:
        @staticmethod
        def get_gain_distance(d, a=1.0):
            return 1 + (1 / (a * d))

        def get_gain_theta(theta, trust_threshold, distrust_threshold):
            theta = np.abs(theta)
            if theta < trust_threshold:
                return 1.0
            elif trust_threshold <= theta and theta < distrust_threshold:
                return np.cos(
                    (np.pi / (2.0 * (distrust_threshold - trust_threshold)))
                    * (theta - trust_threshold)
                )
            else:
                return 0.0

        @staticmethod
        def get_theta(v: np.array, plane: Plane):
            if np.isclose(np.linalg.norm(v), 0):
                return 0.0

            numerator = np.dot(v, plane.n)
            demoninator = np.linalg.norm(v) * np.linalg.norm(plane.n)

            angle = np.arccos(numerator / demoninator)

            angle = angle % (2 * np.pi)

            return angle

        @staticmethod
        def get_intersection_point(p: np.array, v: np.array, plane: Plane):
            denominator = np.dot(plane.n, v)
            numerator = plane.n @ p + plane.d
            if np.isclose(denominator, 0):  # 분모가 0에 가까우면 평행한 상태
                return True, np.array([np.nan, np.nan, np.nan])

            t = -numerator / denominator
            return (t > 0.0 and t < 30.0), p + (t * v)

        @staticmethod
        def get_theta(v: np.array, plane: Plane):
            if np.isclose(np.linalg.norm(v), 0):
                return 0.0

            numerator = np.dot(v, plane.n)
            demoninator = np.linalg.norm(v) * np.linalg.norm(plane.n)

            angle = np.arccos(numerator / demoninator)

            angle = angle % (2 * np.pi)

            return angle

        @staticmethod
        def get_weight_mean_and_covariance(data: np.array, weights: np.array):
            weighted_mean = np.average(data, axis=0, weights=weights)
            centered_data = data - weighted_mean

            weighted_covariance = np.cov(centered_data.T, aweights=weights, bias=True)

            return weighted_mean, weighted_covariance

        @staticmethod
        def get_trust_theta(p: np.array, plane: Plane):
            R = 1.125  # manipulator length
            D = plane.distance(
                np.array([0, 0, 0])
            )  # Box plane과 manipulator base_link 사이 거리
            d = plane.distance(p)  # Box plane과 manipulator eef 사이 거리

            cls = RealIntentionGaussian.IntersectionMethod

            check_eef_intersection, eef_intersection_points = (
                cls.get_intersection_point(p, plane.n, plane)
            )
            check_base_intersection, base_intersection_points = (
                cls.get_intersection_point(np.array([0, 0, 0]), plane.n, plane)
            )

            if check_eef_intersection and check_base_intersection:
                theta_D = np.arccos(D / R)

                theta_d_max = np.arctan(
                    (
                        np.abs(D * np.tan(theta_D))
                        + np.abs(
                            np.linalg.norm(
                                base_intersection_points[1:]
                                - eef_intersection_points[1:]
                            )
                        )
                    )
                    / d
                )
                theta_d_min = np.arctan(
                    (
                        np.abs(D * np.tan(theta_D))
                        - np.abs(
                            np.linalg.norm(
                                base_intersection_points[1:]
                                - eef_intersection_points[1:]
                            )
                        )
                    )
                    / d
                )
                return float(theta_d_min), float(theta_d_max)

            return 0.0, 0.0

    def __init__(self):
        # Map 기준 Plane. TODO: Update plane automatically
        self.plane = Plane(
            # n=np.array([0.99887537, 0.0465492, 0.00900962]), d=-0.8969238154115642
            n=np.array([0.99768449, 0.01408773, 0.0665371]),
            d=-0.9447713826362829,
        )
        self.scale = 1.0
        self.small_value = 0.001

        # Define EEF state
        self.eef_pose_state = State(topic=None, frame_id="VGC")
        # self.eef_twist_state = EEFTwistState(movegroup_name="manipulator")
        self.eef_twist_sub = rospy.Subscriber(
            "/end_effector/twist", Twist, self.eef_twist_callback
        )
        self.eef_twist = Twist()

        self.box_manager = BoxManager(topic="/box_objects/raw")

        # 최종 이차원 평균 및 공분산
        self.mean = np.array([999.0, 999.0])
        self.cov = np.eye(2) * 999.9
        self.distance = float("inf")

        self.pose_with_cov = PoseWithCovarianceStamped()

        # 누적되는 교차점 데이터
        self.intersections = np.empty((0, 3))

        # Control Mode Subscriber
        self.control_mode_sub = rospy.Subscriber(
            "/control_mode", UInt16, self.control_mode_callback
        )
        self.control_mode = URControl.ControlMode.MANUAL

    def control_mode_callback(self, msg: UInt16):
        self.control_mode = URControl.ControlMode(msg.data)

        if self.control_mode == URControl.ControlMode.RESET:
            self.reset()

    def eef_twist_callback(self, msg: Twist):
        """EEF 속도 메세지 콜백"""
        self.eef_twist = msg

    def reset(self):
        """평균, 공분산, 교차점 데이터 초기화"""
        self.mean = np.array([999.0, 999.0])
        self.cov = np.eye(2) * 999.9
        self.intersections = np.empty((0, 3))
        self.distance = float("inf")

    @staticmethod
    def parse_mean_and_cov_to_pose_with_covariance(mean: np.array, cov: np.array):
        if mean.shape != (3,) or cov.shape != (3, 3):
            raise ValueError("Mean must be a 3-element vector and covariance")

        pose_with_cov_stamp = PoseWithCovarianceStamped()

        pose_with_cov_stamp.header = Header(frame_id="map", stamp=rospy.Time.now())
        pose_with_cov_stamp.pose.pose.position = Point(x=mean[0], y=mean[1], z=mean[2])
        pose_with_cov_stamp.pose.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)

        # 6x6 크기의 영행렬 생성
        np_cov = np.zeros((6, 6))

        # 3x3 부분 유지
        np_cov[0:3, 0:3] = cov

        pose_with_cov_stamp.pose.covariance = list(np_cov.flatten())

        return pose_with_cov_stamp

    def calculate_mean_and_cov(self):
        """mean(np.array), cov(np.array) 리턴하는 함수. 자동적으로 내부에 self.mean, self.cov 업데이트, self.intersections 누적. 리턴 무시"""
        p = (
            self.eef_pose_state.get_target_frame_pose()
        )  # Map 프레임 상의 EEF 위치 계산, VGC
        # v = self.eef_twist_state.twist.linear
        v = self.eef_twist.linear

        if p is None:
            rospy.logwarn("No target frame pose")
            return None, None

        test = False

        # TODO: Remove Test Code.

        p: PoseStamped
        if test:
            p = np.array([p.pose.position.x, p.pose.position.y, p.pose.position.z])
            v = np.array(
                [
                    v.x + (random.randint(-100, 100) / 5000),
                    v.y + (random.randint(-100, 100) / 5000),
                    v.z + (random.randint(-100, 100) / 5000),
                ]
            )
        else:
            p = np.array([p.pose.position.x, p.pose.position.y, p.pose.position.z])
            v = np.array([v.x, v.y, v.z])

        self.distance = self.plane.distance(p)

        if np.isclose(np.linalg.norm(v), 0):
            rospy.logwarn("No velocity")
            return False

        # 교점 계산
        forward, intersection = (
            RealIntentionGaussian.IntersectionMethod.get_intersection_point(
                p, v, self.plane
            )
        )

        if np.isnan(intersection).any():
            rospy.logwarn("No intersection")
            return False

        if not forward:
            rospy.logwarn("No forward intersection")
            return False

        try:
            # m -> mm
            intersection *= self.scale
            self.intersections = np.vstack([self.intersections, intersection])

            mean, cov = (
                RealIntentionGaussian.IntersectionMethod.get_weight_mean_and_covariance(
                    self.intersections, np.ones_like(self.intersections[:, 0])
                )
            )

            pose_with_cov = (
                RealIntentionGaussian.parse_mean_and_cov_to_pose_with_covariance(
                    mean, cov
                )
            )

            mean_2d, cov_2d = self.plane.transform_to_2d(mean, cov)

            if mean_2d is not None and cov_2d is not None:
                self.mean = mean_2d
                self.cov = cov_2d
                self.pose_with_cov = pose_with_cov

                return True

        except ValueError as ve:
            # Transform 2D Error
            rospy.logwarn(ve)
            return False

        except Exception as ex:
            rospy.logwarn(ex)
            return False

    def get_boxes_with_pdf(self):
        """박스 매니저에 있는 박스 객체에 PDF를 추가하여 리턴. 박스 매니저 인스턴스를 수정하지는 않음"""
        update_msg = BoxObjectMultiArrayWithPDF()

        # Update header
        update_msg.header = self.box_manager.boxes_msg.header

        if len(self.box_manager.boxes_msg.boxes) == 0:
            # Empty boxes
            rospy.logwarn("get_boxes_with_pdf: No boxes")
            return None

        rv = multivariate_normal(self.mean, self.cov, allow_singular=True)
        max_pdf = rv.pdf(self.mean)

        if max_pdf == 0:
            rospy.logwarn("get_boxes_with_pdf: Max PDF is zero")
            return None

        for box in self.box_manager.boxes_msg.boxes:
            box: BoxObjectWithPDF
            updated_box = box

            box_position = np.array(
                [box.pose.position.x, box.pose.position.y, box.pose.position.z]
            )

            position_2d, _, _ = self.plane.project_to_plane(box_position)

            position_2d *= self.scale

            pdf = rv.pdf(position_2d)

            # Normalize PDF
            updated_box.pdf = np.clip(pdf / max_pdf, self.small_value, 1.0)

            update_msg.boxes.append(box)

        return update_msg


def main():
    rospy.init_node("real_intention_gaussian_node")  # TODO: Add node name

    intention = RealIntentionGaussian()

    # Publisher
    intention_publisher = rospy.Publisher(
        "/box_objects/pdf", BoxObjectMultiArrayWithPDF, queue_size=10
    )

    distance_publisher = rospy.Publisher("/intention/distance", Float32, queue_size=10)

    intersection_publisher = rospy.Publisher(
        "/intention/intersection_length", Int32, queue_size=10
    )

    log_publisher = rospy.Publisher(
        "/intention/log/gaussian", PoseWithCovarianceStamped, queue_size=10
    )

    # rospy.spin()

    r = rospy.Rate(10)  # TODO: Add rate
    while not rospy.is_shutdown():

        # Update mean and covariance
        is_success = intention.calculate_mean_and_cov()

        rospy.loginfo(f"Mean: {intention.mean}")
        rospy.loginfo(f"Num of Intersections: {len(intention.intersections)}")

        # Publish PDF
        if is_success:
            msg = intention.get_boxes_with_pdf()
            if msg is not None:
                # 교차점이 쌓이고, PDF가 계산되면 PDF가 포함된 박스 메세지를 발행
                intention_publisher.publish(msg)

        # 거리, 교차점 수, 로그는 항상 발행
        distance_publisher.publish(Float32(data=intention.distance))
        intersection_publisher.publish(Int32(data=intention.intersections.shape[0]))
        log_publisher.publish(intention.pose_with_cov)

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
