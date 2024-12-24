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
import pcl
from sensor_msgs import point_cloud2


def get_point_cloud(file_path):
    rospy.loginfo("Reading PCD file...")
    return pcl.load_XYZI(file_path)


def parse_point_cloud(pcl_data, frame_id: str, intensity=True, jump=1):
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id  # 원하는 frame_id 설정

    # PointCloud2의 필드 정의 (x, y, z, intensity)
    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("rgb", 12, PointField.UINT32, 1),
    ]

    if not intensity:
        fields.pop()

    # PCL 데이터를 파이썬 리스트 형식으로 변환
    points = []
    for i, point in enumerate(pcl_data):
        # print(point[3])
        if i % jump == 0:
            points.append([point[0], point[1], point[2]])

    # PointCloud2 메시지 생성
    cloud_msg = point_cloud2.create_cloud(header, fields, points)
    return cloud_msg


def main():
    rospy.init_node("joint_state_publisher")  # TODO: Add node name

    point_cloud2_pub = rospy.Publisher("/map", PointCloud2, queue_size=1)

    # PCD 파일 읽기
    dir2 = "/home/irol/vslam_ws/src/pcd_publisher/pcd_file/241108.pcd"
    pcl_data = get_point_cloud(dir2)

    pcl_msg = parse_point_cloud(pcl_data, "map", intensity=False, jump=100)

    point_cloud2_pub.publish(pcl_msg)

    # rospy.spin()

    # return 0

    r = rospy.Rate(1)  # TODO: Add rate
    while not rospy.is_shutdown():
        point_cloud2_pub.publish(pcl_msg)
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
