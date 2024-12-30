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
import cv2
import cv_bridge
import apriltag
import json

try:
    sys.path.append(roslib.packages.get_pkg_dir("localization") + "/src")
    from custom_filter import LowPassFilter
except roslib.exceptions.ROSLibException:
    rospy.logfatal("Cannot find package test_package")
    sys.exit(1)


class AprilTagLocalization:
    def __init__(self):
        root_path = roslib.packages.get_pkg_dir("localization") + "/resources/"
        self.april_tag_detector = AprilTagDetector(
            tag_size=0.04, camera_frame="EEF_camera_link"
        )
        self.april_tag_data = AprilTagData(file_path=root_path + "1220_april_data.json")

        self.user_trigger_subscriber = rospy.Subscriber(
            "/trigger", UInt8, self.trigger_callback
        )
        self.trigger = False

        self.lpf_x = LowPassFilter(cutoff_freq=1.0, ts=0.01)
        self.lpf_y = LowPassFilter(cutoff_freq=1.0, ts=0.01)
        self.lpf_yaw = LowPassFilter(cutoff_freq=1.0, ts=0.01)

        self.transform_data = {
            "time": rospy.Time.now(),
            "parent": "map",
            "child": "base_link_virtual",
            "translation": [0.0, 0.0, 0.0],
            "rotation": (0.0, 0.0, 0.0, 1.0),
        }

        self.tf_broadcaster = tf.TransformBroadcaster()

    def trigger_callback(self, msg: UInt8):
        if msg.data == 0:
            rospy.loginfo("Disabling trigger. Use latest data.")
            self.trigger = False
        elif msg.data == 1:
            rospy.loginfo("Enabling trigger.")
            self.trigger = True
        else:
            rospy.logwarn("Invalid trigger value.")

    def run(self):
        # Detect AprilTags
        local_tags = self.april_tag_detector.tags

        translations = []
        rotations = []

        for tag in local_tags.tags:
            global_pose = self.april_tag_data.get_pose(tag_id=tag.id)
            local_pose = tag.to_pose_stamped()

            if global_pose is None:
                rospy.logwarn(f"Tag {tag.id} not found in the global data.")
                continue

            translation, rotation = self.calculate_tf(
                global_pose=global_pose,
                local_pose=local_pose,
            )

            translations.append(translation)
            rotations.append(rotation)

        if len(translations) == 0 or len(rotations) == 0:
            # If no tags are detected, do not update the transform
            rospy.logwarn("No tags detected.")

        else:
            # If tags are detected and trigger is True, Update the transform data
            if self.trigger:
                avg_translation = np.mean(translations, axis=0).tolist()
                _, _, avg_yaw = np.mean(rotations, axis=0).tolist()

                filtered_x = self.lpf_x.filter(avg_translation[0])
                filtered_y = self.lpf_y.filter(avg_translation[1])
                filtered_yaw = self.lpf_yaw.filter(avg_yaw)

                self.transform_data = {
                    "time": rospy.Time.now(),
                    "parent": "map",
                    "child": "base_link_virtual",
                    "translation": [filtered_x, filtered_y, 0.0],
                    "rotation": quaternion_from_euler(0.0, 0.0, filtered_yaw),
                }

        self.transform_data["time"] = rospy.Time.now()

        self.tf_broadcaster.sendTransform(
            time=self.transform_data["time"],
            parent=self.transform_data["parent"],
            child=self.transform_data["child"],
            translation=self.transform_data["translation"],
            rotation=self.transform_data["rotation"],
        )

        return None

    def calculate_tf(self, local_pose: PoseStamped, global_pose: PoseStamped):
        (local_roll, local_pitch, local_yaw) = euler_from_quaternion(
            [
                local_pose.pose.orientation.x,
                local_pose.pose.orientation.y,
                local_pose.pose.orientation.z,
                local_pose.pose.orientation.w,
            ]
        )

        (global_roll, global_pitch, global_yaw) = euler_from_quaternion(
            [
                global_pose.pose.orientation.x,
                global_pose.pose.orientation.y,
                global_pose.pose.orientation.z,
                global_pose.pose.orientation.w,
            ]
        )

        droll = 0.0  # global_roll - local_roll
        dpitch = 0.0  # global_pitch - local_pitch
        dyaw = global_yaw - local_yaw

        # Calculate translation in 3D space
        trans_x = global_pose.pose.position.x - (
            (local_pose.pose.position.x * np.cos(dyaw) * np.cos(dpitch))
            - (local_pose.pose.position.y * np.sin(dyaw) * np.cos(dpitch))
            - (local_pose.pose.position.z * np.sin(dpitch))
        )

        trans_y = global_pose.pose.position.y - (
            (local_pose.pose.position.x * np.sin(dyaw) * np.cos(droll))
            + (local_pose.pose.position.y * np.cos(dyaw) * np.cos(droll))
            - (local_pose.pose.position.z * np.sin(droll))
        )

        trans_z = 0.0  # global_pose.pose.position.z - local_pose.pose.position.z

        translation = [trans_x, trans_y, trans_z]

        rotation = [droll, dpitch, dyaw]

        return translation, rotation


def main():
    rospy.init_node("april_tag_localization_node")  # TODO: Add node name

    april_tag_localization = AprilTagLocalization()

    r = rospy.Rate(10)  # TODO: Add rate
    while not rospy.is_shutdown():

        april_tag_localization.run()

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
