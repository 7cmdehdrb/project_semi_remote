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


class AprilTagData:
    """Class to load AprilTag data from a JSON file."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = self.load_data()

    def load_data(self):
        f = open(self.file_path, "r")
        return json.load(f)

    def get_pose(self, tag_id: int):
        """Get the pose of a tag with a given ID. If the tag is not found, return None."""
        tags_data = self.data["tags"]

        for tag in tags_data:
            if int(tag["id"]) == tag_id:
                position = tag["pose"]["position"]
                orientation = tag["pose"]["orientation"]

                pose = PoseStamped()
                pose.header = Header(frame_id="map", stamp=rospy.Time.now())

                pose.pose.position.x = position["x"]
                pose.pose.position.y = position["y"]
                pose.pose.position.z = position["z"]

                pose.pose.orientation.x = orientation["x"]
                pose.pose.orientation.y = orientation["y"]
                pose.pose.orientation.z = orientation["z"]
                pose.pose.orientation.w = orientation["w"]

                return pose

        return None


class AprilTagDetector:
    """Class to detect AprilTags in the camera feed."""

    class AprilTag:
        """Class to store AprilTag data."""

        def __init__(self, id: int, pose: Pose):
            self.id = id
            self.pose = pose

        def to_json(self):
            return {
                "id": self.id,
                "pose": {
                    "position": {
                        "x": self.pose.position.x,
                        "y": self.pose.position.y,
                        "z": self.pose.position.z,
                    },
                    "orientation": {
                        "x": self.pose.orientation.x,
                        "y": self.pose.orientation.y,
                        "z": self.pose.orientation.z,
                        "w": self.pose.orientation.w,
                    },
                },
            }

        def to_pose_stamped(self, frame_id: str = "base_link"):
            pose_stamped = PoseStamped()
            pose_stamped.header = Header(frame_id=frame_id, stamp=rospy.Time.now())
            pose_stamped.pose = self.pose
            return pose_stamped

    class AprilTagArray:
        """Class to store multiple AprilTags."""

        def __init__(self, header: Header, tags: list):
            self.header = header
            self.tags = tags

        def append(self, tag):
            self.tags.append(tag)

        def to_json(self):
            return {"tags": [tag.to_json() for tag in self.tags]}

    def __init__(self, tag_size: float = 0.04, camera_frame: str = "EEF_camera_link"):
        # Initialize the AprilTag detector
        self.bridge = cv_bridge.CvBridge()
        self.tag_detector = apriltag.Detector(
            options=apriltag.DetectorOptions(families="tag36h11")
        )

        # Initialize the camera frame
        self.base_frame = "base_link"
        self.camera_frame = camera_frame

        self.object_points = np.array(
            [
                [-tag_size / 2.0, -tag_size / 2.0, 0.0],
                [tag_size / 2.0, -tag_size / 2.0, 0.0],
                [tag_size / 2.0, tag_size / 2.0, 0.0],
                [-tag_size / 2.0, tag_size / 2.0, 0.0],
            ]
        )

        # ROS
        self.tf_listener = tf.TransformListener()

        self.color_image_subscriber = rospy.Subscriber(
            "/d405_camera/color/image_raw", Image, self.color_image_callback
        )

        self.camera_info = None
        self.camera_info_subscriber = rospy.Subscriber(
            "/d405_camera/color/camera_info", CameraInfo, self.camera_info_callback
        )

        # Initialize AprilTagArray
        self.tags = self.AprilTagArray(header=Header(frame_id=self.base_frame), tags=[])

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info = msg

    def color_image_callback(self, msg: Image):
        """Callback function to process the color image."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv_image_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            self.tags = self.detect_april_tags(cv_image_gray)

        except cv_bridge.CvBridgeError as cv_bridge_ex:
            rospy.logerr(cv_bridge_ex)
        except Exception as ex:
            rospy.logerr(ex)

    def parse_transform(self, translation_vector: list, rotation_vector: list):
        pose_stamped = PoseStamped()

        quaternion = quaternion_from_euler(
            rotation_vector[0], rotation_vector[1], rotation_vector[2]
        )

        pose_stamped.header = Header(frame_id=self.camera_frame, stamp=rospy.Time(0))
        pose_stamped.pose = Pose(
            position=Point(
                x=translation_vector[0],
                y=translation_vector[1],
                z=translation_vector[2],
            ),
            orientation=Quaternion(
                x=quaternion[0],
                y=quaternion[1],
                z=quaternion[2],
                w=quaternion[3],
            ),
        )

        return pose_stamped

    def detect_april_tags(self, np_image: np.array):
        """Detect AprilTags in the image. Input grayscale image and return AprilTagArray"""
        detected_tag = self.tag_detector.detect(np_image)

        if self.camera_info is None:
            rospy.logwarn("Camera info is not available.")
            return self.AprilTagArray(header=Header(frame_id="base_link"), tags=[])

        # Camera matrix and distortion coefficients
        camera_matrix = np.array(self.camera_info.K).reshape(3, 3)
        dist_coeffs = np.array(self.camera_info.D)

        # Initialize AprilTagArray (return value)
        tag_array = self.AprilTagArray(
            header=Header(frame_id=self.camera_frame, stamp=rospy.Time.now()), tags=[]
        )

        #  detected_tag type is apriltag.Detection()
        for tag in detected_tag:
            image_points = np.array(tag.corners)

            # Solve PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.object_points, image_points, camera_matrix, dist_coeffs
            )

            if not success:
                rospy.logwarn("Failed to solve PnP.")
                return tag_array

            # Pose of the tag in the camera frame
            translation_vector = translation_vector.flatten()
            rotation_vector = rotation_vector.flatten()

            tag_posestamped_camera_frame = self.parse_transform(
                translation_vector, rotation_vector
            )

            # Transform the pose to the base frame
            if self.tf_listener.canTransform(
                self.base_frame, self.camera_frame, rospy.Time(0)
            ):
                transformed_pose = self.tf_listener.transformPose(
                    self.base_frame, tag_posestamped_camera_frame
                )

                # Append the tag to the tag_array
                tag_array.append(
                    self.AprilTag(id=tag.tag_id, pose=transformed_pose.pose)
                )

        return tag_array


class AprilTagLocalization:
    def __init__(self):
        root_path = roslib.packages.get_pkg_dir("localization") + "/resources/"
        self.april_tag_detector = AprilTagDetector(
            tag_size=0.04, camera_frame="EEF_camera_link"
        )
        self.april_tag_data = AprilTagData(file_path=root_path + "1113.json")

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
            "child": "base_link",
            "translation": [0.0, 0.0, 0.0],
            "rotation": (0.0, 0.0, 0.0, 1.0),
        }

        self.tf_broadcaster = tf.TransformBroadcaster()

    def trigger_callback(self, msg: UInt8):
        if msg.data == 0:
            self.trigger = False
        elif msg.data == 1:
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
            avg_translation = [0.0, 0.0, 0.0]
            avg_yaw = 0.0

            rospy.logwarn("No tags detected.")

        else:
            avg_translation = np.mean(translations, axis=0).tolist()
            _, _, avg_yaw = np.mean(rotations, axis=0).tolist()

        filtered_x = self.lpf_x.filter(avg_translation[0])
        filtered_y = self.lpf_y.filter(avg_translation[1])
        filtered_yaw = self.lpf_yaw.filter(avg_yaw)

        # If trigger is True, Update the transform data
        if self.trigger:
            self.transform_data = {
                "time": rospy.Time.now(),
                "parent": "map",
                "child": "base_link",
                "translation": [filtered_x, filtered_y, 0.0],
                "rotation": quaternion_from_euler(0.0, 0.0, filtered_yaw),
            }

        # If trigger is False, Use the previous transform data, Only Update the time
        else:
            self.transform_data["time"] = rospy.Time.now()

        self.tf_broadcaster.sendTransform(
            time=self.transform_data["time"],
            parent=self.transform_data["parent"],
            child=self.transform_data["child"],
            translation=self.transform_data["translation"],
            rotation=self.transform_data["rotation"],
        )

        return avg_translation, filtered_yaw

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

        trans, yaw = april_tag_localization.run()

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
