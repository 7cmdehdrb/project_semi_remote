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

# Custom modules
import cv2
import cv_bridge
import apriltag
import json


class PositionAverageFilter:
    def __init__(self):
        self.i = 0
        self.average = np.array([0.0, 0.0, 0.0])

    def filter(self, data, enable=True):
        if not enable:
            return data
        # 샘플 수 +1 (+1 the number of sample)
        self.i += 1

        # 평균 필터의 alpha 값 (alpha of average filter)
        alpha = (self.i - 1) / (self.i + 0.0)

        # 평균 필터의 재귀식 (recursive expression of average filter)
        average = alpha * self.average + (1 - alpha) * data

        # 평균 필터의 이전 상태값 업데이트 (update previous state value of average filter)
        self.average = average

        return average


class AprilBoxPublisher:
    def __init__(self):
        self.april_detector = AprilTagDetector(
            camera_frame="EEF_camera_link", tag_size=0.04
        )

        self.boxes = None
        self.filters = {}

        self.april_box_pub = rospy.Publisher(
            "/box_objects/raw", BoxObjectMultiArrayWithPDF, queue_size=1
        )
        self.april_pose_pub = rospy.Publisher(
            "/box_objects/pose_array", PoseArray, queue_size=1
        )

    def parse_box_object_to_posearray(self) -> PoseArray:
        pose_array = PoseArray()

        if self.boxes is None:
            return pose_array

        pose_array.header = self.boxes.header

        for box in self.boxes.boxes:
            pose = Pose(
                position=box.pose.position,
                orientation=Quaternion(
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    w=1.0,
                ),
            )
            # pose_array.poses.append(box.pose)
            pose_array.poses.append(pose)

        return pose_array

    def publish(self):
        # Initialize the self.boxes
        if self.boxes is None:
            for tag in self.april_detector.tags.tags:
                tag: BoxObjectWithPDF
                self.filters[str(tag.id)] = PositionAverageFilter()

            new_box_array = BoxObjectMultiArrayWithPDF()
            new_box_array.header = Header(frame_id="map", stamp=rospy.Time.now())

            for tag in self.april_detector.tags.tags:
                tag: BoxObjectWithPDF

                avg_filter = self.filters[str(tag.id)]
                avg_filter: PositionAverageFilter

                average = avg_filter.filter(
                    np.array(
                        [
                            tag.pose.position.x,
                            tag.pose.position.y,
                            tag.pose.position.z,
                        ]
                    ),
                    enable=False,
                )

                pose = Pose(
                    position=Point(
                        x=average[0],
                        y=average[1],
                        z=average[2],
                    ),
                    orientation=tag.pose.orientation,
                )

                tag.pose = pose

                new_box_array.boxes.append(tag)

            self.boxes = new_box_array
            self.april_box_pub.publish(self.boxes)

            return

        new_box_array = self.boxes

        existed_id = [box.id for box in self.boxes.boxes]
        new_id = [tag.id for tag in self.april_detector.tags.tags]

        # 새로 인식된 ID (Newly recognized ID)
        newly_recognized_id = set(new_id) - set(existed_id)

        for tag in self.april_detector.tags.tags:
            if tag.id in newly_recognized_id:
                # 새로운 박스를 추가 (Add new box)
                avg_filter = PositionAverageFilter()
                self.filters[str(tag.id)] = avg_filter

                new_box = BoxObjectWithPDF()

                new_box.id = tag.id
                new_box.header = Header(frame_id="map", stamp=rospy.Time.now())

                average = avg_filter.filter(
                    np.array(
                        [
                            tag.pose.position.x,
                            tag.pose.position.y,
                            tag.pose.position.z,
                        ]
                    ),
                    enable=False,
                )

                pose = Pose(
                    position=Point(
                        x=average[0],
                        y=average[1],
                        z=average[2],
                    ),
                    orientation=tag.pose.orientation,
                )

                new_box.pose = pose

                new_box_array.boxes.append(new_box)

            else:
                # 기존 박스의 위치 업데이트 (Update the position of the existing box)
                for box in self.boxes.boxes:
                    if box.id == tag.id:
                        avg_filter = self.filters[str(tag.id)]
                        avg_filter: PositionAverageFilter

                        average = avg_filter.filter(
                            np.array(
                                [
                                    tag.pose.position.x,
                                    tag.pose.position.y,
                                    tag.pose.position.z,
                                ]
                            ),
                            enable=False,
                        )

                        pose = Pose(
                            position=Point(
                                x=average[0],
                                y=average[1],
                                z=average[2],
                            ),
                            orientation=tag.pose.orientation,
                        )

                        box.pose = pose

        # self.boxes 인스턴스 업데이트
        self.boxes = new_box_array

        rospy.loginfo(f"Publishing {len(self.boxes.boxes)} boxes.")

        # Publish the boxes
        self.april_box_pub.publish(self.boxes)
        self.april_pose_pub.publish(self.parse_box_object_to_posearray())


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

    def __init__(
        self,
        tag_size: float = 0.04,
        camera_frame: str = "EEF_camera_link",
        target_frame=None,
    ):
        # Initialize the AprilTag detector
        self.bridge = cv_bridge.CvBridge()
        self.tag_detector = apriltag.Detector(
            options=apriltag.DetectorOptions(families="tag36h11")
        )

        # Initialize the camera frame
        self.base_frame = "base_link"
        self.camera_frame = camera_frame
        self.target_frame = target_frame

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

            self.tags = self.detect_april_tags(
                cv_image_gray, target_frame=self.target_frame
            )

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

    def detect_april_tags(self, np_image: np.array, target_frame: str):
        """Detect AprilTags in the image. Input grayscale image and return AprilTagArray"""
        if target_frame is None:
            target_frame = self.base_frame

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

            # Transform the pose to the target frame
            if self.tf_listener.canTransform(
                target_frame, self.camera_frame, rospy.Time(0)
            ):
                transformed_pose = self.tf_listener.transformPose(
                    target_frame, tag_posestamped_camera_frame
                )

                # Append the tag to the tag_array
                tag_array.append(
                    self.AprilTag(id=tag.tag_id, pose=transformed_pose.pose)
                )

        return tag_array


def main():
    rospy.init_node("april_box_publisher_node")  # TODO: Add node name

    april_publisher = AprilBoxPublisher()

    r = rospy.Rate(10)  # TODO: Add rate
    while not rospy.is_shutdown():
        april_publisher.publish()
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
