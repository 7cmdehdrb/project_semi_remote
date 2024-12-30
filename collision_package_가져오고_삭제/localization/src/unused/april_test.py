import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_system_default
import numpy as np
from std_msgs.msg import *
from sensor_msgs.msg import *
from geometry_msgs.msg import *
import cv2
import cv_bridge
import apriltag


def euler_from_quaternion(quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def quaternion_from_euler(roll, pitch, yaw):
    """
    Converts euler roll, pitch, yaw to quaternion (w in last place)
    quat = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return qx, qy, qz, qw


class AprilTest(Node):
    def __init__(self):
        super().__init__("april_test_node")

        self.use_gui = True

        self.bridge = cv_bridge.CvBridge()
        self.tag_detector = apriltag.Detector(
            options=apriltag.DetectorOptions(families="tag36h11")
        )

        self.image_array = None
        self.image_sub = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.image_callback,
            qos_profile_system_default,
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self.camera_info_callback,
            qos_profile_system_default,
        )
        self.camera_info = CameraInfo()

        self.tag_pose_pub = self.create_publisher(
            PoseStamped, "/tag_pose", qos_profile_system_default
        )

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info = msg

    def image_callback(self, msg: Image):
        # Convert ROS Image message to OpenCV format (NumPy array)
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv_image_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            self.image_array = cv_image

            if self.use_gui:
                cv2.imshow("Image", cv_image)
                cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

        self.detect_april(cv_image_gray)

    def detect_april(self, image: np.array):
        results = self.tag_detector.detect(image)

        camera_matrix = np.array(self.camera_info.k).reshape(3, 3)
        dist_coeffs = np.array(self.camera_info.d)

        # 태그 크기 정의 (예: 태그 한 변의 길이가 0.05m인 경우)
        tag_size = 0.04

        # 월드 좌표계에서 태그 코너 위치 정의
        object_points = np.array(
            [
                [-tag_size / 2, -tag_size / 2, 0],
                [tag_size / 2, -tag_size / 2, 0],
                [tag_size / 2, tag_size / 2, 0],
                [-tag_size / 2, tag_size / 2, 0],
            ]
        )

        for detection in results:
            image_points = np.array(detection.corners)
            success, rotation_vector, translation_vector = cv2.solvePnP(
                object_points, image_points, camera_matrix, dist_coeffs
            )

            if success:
                # 월드 좌표계에서 태그 위치 출력
                self.get_logger().info(f"Tag ID: {detection.tag_id}")
                self.get_logger().info(f"Translation: {translation_vector}")
                self.get_logger().info(f"Rotation: {rotation_vector}")

                translation_matrix = translation_vector.tolist()
                rotation_matrix = rotation_vector.tolist()

                msg = PoseStamped()

                msg.header.frame_id = "base_link"
                msg.header.stamp = Time().to_msg()

                msg.pose.position.x = translation_matrix[0][0]
                msg.pose.position.y = translation_matrix[1][0]
                msg.pose.position.z = translation_matrix[2][0]

                quat = quaternion_from_euler(
                    rotation_matrix[0][0], rotation_matrix[1][0], rotation_matrix[2][0]
                )

                msg.pose.orientation.x = quat[0]
                msg.pose.orientation.y = quat[1]
                msg.pose.orientation.z = quat[2]
                msg.pose.orientation.w = quat[3]

                self.tag_pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = AprilTest()

    rclpy.spin(node)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
