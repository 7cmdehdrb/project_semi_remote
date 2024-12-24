#!/usr/bin/python3

import os
import sys
import roslib.exceptions
import roslib.packages
import rospy
import roslib
import numpy as np
import cv2
from abc import ABC, abstractmethod

# Messages
from std_msgs.msg import *
from sensor_msgs.msg import *
from geometry_msgs.msg import *
from nav_msgs.msg import *
from custom_msgs.msg import DifferentialEncoder

# Custom
try:
    sys.path.append(roslib.packages.get_pkg_dir("test_package") + "/src")
    from sensors import Encoder as AbstractEncoder
    from sensors import VIO as AbstractVIO
except roslib.exceptions.ROSLibException:
    rospy.logfatal("Cannot find package test_package")
    sys.exit(1)


class Encoder(AbstractEncoder):
    def __init__(self, topic):
        super().__init__(topic)

        # Reinitialize the H matrix
        pass  # Same as default value

        # Reinitialize the P matrix
        self.P = np.diag([0.1] * 9)

        # Observers
        self.register_observer(self.parse_encoder)

        # Parameters
        self.wheel_base = (
            1.0  # Distance between the two wheels. TODO: Change this value
        )

        # Local variables
        self.last_time = rospy.Time.now()

        # ROS
        self.subscription = rospy.Subscriber(topic, DifferentialEncoder, self.update)

    def update(self, msg: DifferentialEncoder):
        return super().update(msg)

    # Registered method to update x and P via encoder data
    def parse_encoder(self):
        # Get time information
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        # Get the velocity of each wheel
        left_velocity = self.data.left_wheel
        right_velocity = self.data.right_wheel
        linear_velocity = (left_velocity + right_velocity) / 2.0

        # Compute the distance traveled by each wheel
        left_distance = left_velocity * dt
        right_distance = right_velocity * dt

        # Compute the change in orientation
        delta_theta = (right_distance - left_distance) / self.wheel_base

        # Compute the average distance traveled
        delta_distance = (left_distance + right_distance) / 2.0

        # Update x and y using the Kalman filter state

        self.x[0][0] += delta_distance * np.cos(self.x[8][8] + delta_theta / 2.0)
        self.x[1][1] += delta_distance * np.sin(self.x[8][8] + delta_theta / 2.0)
        self.x[2][2] = 0.0

        self.x[3][3] = linear_velocity * np.cos(self.x[8][8])
        self.x[4][4] = linear_velocity * np.sin(self.x[8][8])
        self.x[5][5] = 0.0

        self.x[6][6] = 0.0
        self.x[7][7] = 0.0
        self.x[8][8] += delta_theta

        # Normalize theta to be within the range [-pi, pi]
        self.x[8][8] = (self.theta + np.pi) % (2 * np.pi) - np.pi


class VIO(AbstractVIO):
    def __init__(self, topic):
        super().__init__(topic)

        # Reinitialize the H matrix
        pass  # Same as default value

        # Reinitialize the P matrix
        self.P = np.diag([0.1] * 9)

        # Observers
        self.register_observer(self.parse_vio)

        # Parameters
        self.trans = [0.0, 0.0, 0.0]
        self.rot = [0.0, 0.0, 0.0]

        # ROS
        self.subscription = rospy.Subscriber(topic, Odometry, self.update)

    def update(self, msg: Odometry):
        return super().update(msg)

    # calculate T and R
    def get_transform_matrix(self, translation, rotation):
        # Rotation 행렬 생성 (Z-Y-X 순서)

        x = translation[0]
        y = translation[1]
        z = translation[2]

        roll = rotation[0]
        pitch = rotation[1]
        yaw = rotation[2]

        Rz = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )

        Ry = np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)],
            ]
        )

        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)],
            ]
        )

        # 최종 회전 행렬 (Rz * Ry * Rx)
        R = Rz @ Ry @ Rx

        # 변환 행렬 생성 (4x4)
        T = np.eye(4)
        T[:3, :3] = R  # 회전 부분 설정
        T[:3, 3] = [x, y, z]  # 이동 부분 설정

        return T, R  # 변환 행렬과 회전 행렬 반환

    def apply_transform(self, T, R):
        # Extract the point array from the data array
        point = np.diag(self.x)[:3]  # [x, y, z]
        homogeneous_point = np.append(point, 1)  # [x, y, z, 1]

        # 변환 행렬을 적용
        transformed_point = (T @ homogeneous_point)[:3]  # [x, y, z]

        velocity = np.diag(self.data)[3:6]  # [vx, vy, vz]
        rotated_velocity = R @ velocity

        orientation = np.diag(self.data)[6:9]  # [r, p, y]
        rotated_orientation = R @ orientation

        # [x, y, z, vx, vy, vz, r, p ,y]
        merged_x = np.concatenate(
            (transformed_point, rotated_velocity, rotated_orientation)
        )

        # Change marged_x to a diagonal matrix and Update the x matrix
        self.x = np.diag(merged_x)

    def parse_vio(self):
        T, R = self.get_transform_matrix(self.trans, self.rot)

        # Apply the transformation to the state
        self.apply_transform(T, R)


class Kalman(object):
    def __init__(self, sensors: list):
        self.x = np.zeros((9, 9))  # state matrix: [x, y, z, vx, vy, vz, r, p ,y]
        self.A = np.eye(9)  # model matrix
        self.P = self.P = np.diag([0.1] * 9)  # error covariance matrix
        self.Q = np.diag([0.1] * 9)  # uncertainty matrix
        self.H = np.zeros((9, 9)).astype(int)

        self.sensors = sensors

    def filter(self):
        # Calculate the error covariance
        R = np.zeros((9, 9))
        for sensor in self.sensors:
            R += sensor.P

        # Update the state estimate
        x_k = self.A @ self.x

        # Update the error covariance matrix
        P_k = self.A @ R @ self.A.T + self.Q

        # Calculate each sensor's & total Kalman gain
        K = np.zeros((9, 9))
        for sensor in self.sensors:
            s = sensor.H @ sensor.P @ self.H.T + self.P
            k = P_k @ self.H.T @ np.linalg.inv(s)

            K += k @ np.linalg.inv(s)
            sensor.K = k

        # Update the state estimate
        for sensor in self.sensors:
            x_u = sensor.K @ (sensor.x - sensor.H @ x_k)
            x_k += x_u

        # Update final state and error covariance
        self.x = x_k
        self.P = P_k - K @ self.H @ P_k

        return self.x, self.P


def main():
    rospy.init_node("kalman_filter_node")

    encoder = Encoder("encoder")
    vio = VIO("vio")

    sensors = [encoder, vio]

    kalman = Kalman(sensors=sensors)

    r = rospy.Rate(30)
    while not rospy.is_shutdown():
        x, P = kalman.filter()
        r.sleep()


if __name__ == "__main__":
    main()
