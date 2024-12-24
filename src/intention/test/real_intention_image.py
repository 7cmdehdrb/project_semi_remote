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

# Messages
from std_msgs.msg import *
from sensor_msgs.msg import *
from geometry_msgs.msg import *
from nav_msgs.msg import *

# Custom
try:
    sys.path.append(roslib.packages.get_pkg_dir("test_package") + "/src")
    from abstract_real_intention_image import FState as AbstractFState
    from abstract_real_intention_image import FOV as AbstractFOV
    from abstract_real_intention_image import RealIntention as AbstractRealIntention
    from sensors import RealSenseD455_D, RealSenseD455_RGB
except roslib.exceptions.ROSLibException:
    rospy.logfatal("Cannot find package test_package")
    sys.exit(1)


class EndEffectorState(AbstractFState):
    def __init__(self, topic):
        super().__init__(topic)

        self.last_time = rospy.Time.now()
        self.last_odom = Odometry()

        self.odom_sub = rospy.Subscriber(topic, Odometry, self.odom_callback)
        self.test_pub = rospy.Publisher("/test", Vector3, queue_size=10)

        self.register_observer(self.calculate_covariance)

    # observer pattern
    def register_observer(self, observer_callback):
        return super().register_observer(observer_callback)

    # observer pattern
    def notify_observers(self):
        return super().notify_observers()

    # odometry callback
    def odom_callback(self, msg: Odometry):
        """This callback function input nav_msgs.msg.Odometry and update robot's state"""
        if not isinstance(msg, Odometry):
            rospy.logerr("Received message is not of type Odometry")
            return

        # Logic
        self.data = msg  # Update data

        self.notify_observers()  # call all observers

        self.last_odom = self.data

    def calculate_covariance(self):
        """Automatically called after odom_callback"""
        pass

    def calculate_linear_acceleration(self):
        """Automatically called after odom_callback"""
        # Logic
        # Calculate delta time

        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        dt = 0.1

        # Update linear acceleration
        current_linear_velocity = self.data.twist.twist.linear
        last_linear_velocity = self.last_odom.twist.twist.linear

        x = (current_linear_velocity.x - last_linear_velocity.x) / dt
        y = (current_linear_velocity.y - last_linear_velocity.y) / dt
        z = (
            np.linalg.norm([current_linear_velocity.x, current_linear_velocity.y])
            - np.linalg.norm([last_linear_velocity.x, last_linear_velocity.y])
        ) / dt
        size = np.linalg.norm([x, y])

        self.linear_acceleration = Vector3(x=x, y=y, z=z)

        # Update last time
        self.last_time = current_time


class FOV(AbstractFOV):
    def __init__(self, state, camera, depth_camera):
        super().__init__(state, camera, depth_camera)

        self.bridge = CvBridge()

        self.fx, self.fy = 525.0, 525.0  # 초점 거리 (픽셀 단위)
        self.dy = 0.3  # 그리퍼 - 카메라 거리 (m)

        self.target_pixel_x = 0
        self.target_pixel_y = 0

        # self.camera.register_observer()
        self.depth_camera.register_observer(self.calculate_target_pixel)

    def calculate_target_pixel(self):
        cx, cy = (
            (self.depth_camera.data.width / 2.0) - 0.5,
            (self.depth_camera.data.height / 2.0) - 0.5,
        )

        # Calculate center

        extracted_row = self.depth_camera.data[
            int(self.depth_camera.data.width / 2.0),
            int(self.depth_camera.data.height / 2.0) :,
        ]

        min_y = float("inf")
        min_i = -1

        for i, z in enumerate(extracted_row):
            y = (i - cy) * z / self.fy
            dy = abs(y - self.dy)

            if dy < min_y:
                min_y = dy
                min_i = i

            if min_y < 0.01:
                break

        self.target_pixel_x = int(self.camera.data.width / 2.0)
        self.target_pixel_y = int(self.camera.data.height / 2.0) + min_i

    def project_image(self, image):
        # Create a blank image
        image_height, image_width = (
            self.depth_camera.data.height,
            self.depth_camera.data.width,
        )
        blank_image = np.zeros((image_height, image_width, 3), np.uint8)

        # Define the circle parameters
        center_coordinates = (self.target_pixel_x, self.target_pixel_y)
        radius = 20  # TODO: Change this value from covariance
        color = (255, 0, 0)  # Red color in BGR
        thickness = -1  # Fill the circle

        # Draw the circle on the blank image
        cv2.circle(blank_image, center_coordinates, radius, color, thickness)

        # Overlay the circle image on the original image
        alpha = 0.5  # Transparency factor
        cv2.addWeighted(blank_image, alpha, image, 1 - alpha, 0, image)

        # Convert the OpenCV image (blank_image) to a ROS Image message
        return self.bridge.cv2_to_imgmsg(blank_image, encoding="bgr8")

    def check_overlay(self, target_object):
        pass


class RealIntentionImage(AbstractRealIntention):
    def __init__(self, state, target_objects):
        super().__init__(state, target_objects)

    def check_best_target_object(self):
        pass

    def run(self):
        pass


def main():
    rospy.init_node("real_intention_image_node")

    # end_effector_state = EndEffectorState(topic="/odometry/end_effector")
    # fov = FOV(end_effector_state)
    pub = rospy.Publisher("/real_intention_image", Image, queue_size=10)

    # Read the image file
    image_path = (
        "/home/catkin_ws/src/intention/resource/959dab9ab97b3d3011121fcd3300f861.jpg"
    )
    image = np.array(cv2.imread(image_path, cv2.IMREAD_COLOR))

    bridge = CvBridge()

    # Create a blank image
    image_height, image_width = (1920, 1080)
    blank_image = np.zeros((image_width, image_height, 3), np.uint8)

    # Define the circle parameters
    center_coordinates = (960, 540)
    radius = 540  # TODO: Change this value from covariance
    color = (255, 0, 0)  # Red color in BGR
    thickness = -1  # Fill the circle

    # Draw the circle on the blank image
    cv2.circle(blank_image, center_coordinates, radius, color, thickness)

    # Overlay the circle image on the original image
    alpha = 0.5  # Transparency factor
    cv2.addWeighted(blank_image, alpha, image, 1, 0, image)

    # Convert the OpenCV image (blank_image) to a ROS Image message
    msg = bridge.cv2_to_imgmsg(image, encoding="rgb8")

    r = rospy.Rate(1)
    # return
    while not rospy.is_shutdown():
        pub.publish(msg)
        r.sleep()


if __name__ == "__main__":
    rospy.init_node("real_intention_image_node")

    end_effector = EndEffectorState(topic="/odometry/end_effector")

    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        end_effector.test_pub.publish(end_effector.linear_acceleration)
        if (
            end_effector.linear_acceleration.x != 0.0
            or end_effector.linear_acceleration.y != 0.0
            or end_effector.linear_acceleration.z != 0.0
        ):
            # print(end_effector.linear_acceleration)
            # print(end_effector.data.twist.twist.linear)
            # print("\n")
            pass
        r.sleep()
