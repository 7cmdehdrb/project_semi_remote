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
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.integrate import quad
from enum import Enum

try:
    sys.path.append(roslib.packages.get_pkg_dir("base_package") + "/src")
    from state import State, EEFTwistState
except roslib.exceptions.ROSLibException:
    rospy.logfatal("Cannot find package test_package")
    sys.exit(1)


class Plane(object):
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
    def __init__(self):
        self.boxes_subscriber = rospy.Subscriber(
            "", BoxObjectMultiArrayWithPDF, self.boxes_callback
        )
        
    def boxes_callback(self, msg: BoxObjectMultiArrayWithPDF):
        for box in msg.boxes:
            # Do something with the box
            pass


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
        # Define EEF
        self.eef_pose_state = State(topic=None, frame_id="VGC")
        self.eef_twist_state = EEFTwistState()

        # 최종 이차원 평균 및 공분산
        self.mean = np.array([0, 0])
        self.cov = np.eye(2)

        # 누적되는 교차점 데이터
        self.intersections = np.empty((0, 3))

        self.plane = Plane(
            n=np.array([0.99887537, 0.0465492, 0.00900962]), d=-0.8969238154115642
        )

    def reset(self):
        self.mean = np.array([0, 0])
        self.cov = np.eye(2)
        self.intersections = np.empty((0, 3))

    def calculate_mean_and_cov(self):
        """mean(np.array), cov(np.array) 리턴하는 함수. 자동적으로 내부에 self.intersections 누적"""
        p = (
            self.eef_pose_state.get_target_frame_pose()
        )  # Map 프레임 상의 EEF 위치 계산, VGC
        v = self.eef_twist_state.twist.linear

        if p is None:
            rospy.logwarn("No target frame pose")
            return None, None

        p: PoseStamped
        p = np.array([p.pose.position.x, p.pose.position.y, p.pose.position.z])
        v = np.array([v.x, v.y, v.z])

        # 교점 계산
        _, intersection = (
            RealIntentionGaussian.IntersectionMethod.get_intersection_point(
                p, v, self.plane
            )
        )

        # 초기화
        mean_2d, cov_2d = None, None

        if np.isnan(intersection).any():
            # rospy.logwarn("No intersection")
            return None, None

        try:
            self.intersections = np.vstack([self.intersections, intersection])

            mean, cov = (
                RealIntentionGaussian.IntersectionMethod.get_weight_mean_and_covariance(
                    self.intersections, np.ones_like(self.intersections[:, 0])
                )
            )

            # 3D to 2D
            mean_2d, cov_2d = self.plane.transform_to_2d(mean, cov)

            return mean_2d, cov_2d

        except ValueError as ve:
            # Transform 2D Error
            rospy.logwarn(ve)
            return None, None

        except Exception as ex:
            rospy.logwarn(ex)
            return None, None


def main():
    rospy.init_node("real_intention_gaussian_node")  # TODO: Add node name

    intention = RealIntentionGaussian()

    r = rospy.Rate(10)  # TODO: Add rate
    while not rospy.is_shutdown():
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
