#!/usr/bin/python3

# Basic modules
import os
import sys
import rospy
import roslib
import numpy as np
import math as m

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


def parse_ascii_pcd(file_path):
    with open(file_path, "r") as f:
        # 헤더 파싱
        header = []
        for line in f:
            header.append(line.strip())
            if line.startswith("DATA"):
                break

        # 데이터 부분 읽기
        data = np.loadtxt(f)

    print("Header:", header)
    print("Data sample (first 10 points):\n", data[:10])


def get_point_cloud(file_path):
    rospy.loginfo("Reading PCD file...")
    return pcl.load_XYZRGB(file_path)


def parse_point_cloud(pcl_data, frame_id: str, jump=1):
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id  # 원하는 frame_id 설정

    # PointCloud2의 필드 정의 (x, y, z, rgb)
    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("rgb", 12, PointField.UINT32, 1),  # RGB를 하나의 UINT32 필드로 추가
    ]

    # PCL 데이터를 파이썬 리스트 형식으로 변환
    points = []
    for i, point in enumerate(pcl_data):
        if i % jump == 0:
            x, y, z = (
                point[0],
                point[1],
                point[2],
            )

            rgb = point[3]

            if m.isnan(rgb):
                continue

            rgb = int(rgb)

            r = (rgb >> 16) & 0xFF
            g = (rgb >> 8) & 0xFF
            b = rgb & 0xFF
            rgb = (r << 16) | (g << 8) | b

            # print(f"R: {r}, G: {g}, B: {b}")

            points.append([x, y, z, rgb])

    # PointCloud2 메시지 생성
    cloud_msg = point_cloud2.create_cloud(header, fields, points)
    return cloud_msg


def print_raw_point_cloud(file_path, sample_size=10):
    # PCD 파일 로드
    pcd = pcl.load(file_path)

    # numpy 배열로 변환하여 구조 확인
    point_array = pcd.to_array()

    print(f"Total points in the cloud: {len(point_array)}")

    # 지정한 샘플 크기만큼 원시 포인트 클라우드 데이터를 출력
    for i, point in enumerate(point_array[:sample_size]):
        x, y, z = point[0], point[1], point[2]
        rgb = int(point[3]) if len(point) > 3 else None
        normal_x = point[4] if len(point) > 4 else None
        normal_y = point[5] if len(point) > 5 else None
        normal_z = point[6] if len(point) > 6 else None
        curvature = point[7] if len(point) > 7 else None

        if rgb is not None:
            print(f"Point {i}: ({x}, {y}, {z}), RGB: {rgb}")


def main():
    rospy.init_node("pointcloud_publisher")  # 노드 이름을 "pointcloud_publisher"로 설정

    point_cloud2_pub = rospy.Publisher("/map", PointCloud2, queue_size=1)

    # PCD 파일 읽기
    dir2 = "/home/catkin_ws/src/localization/resources/global_240516_rejectedLoopclosure.pcd"
    dir2 = "/home/catkin_ws/src/localization/resources/global_240517_rejectedLoopclosure.pcd"

    # 예시 파일 경로에 맞춰 사용하세요
    parse_ascii_pcd(dir2)

    return

    pcl_data = get_point_cloud(dir2)

    pcl.load_XYZRGB

    pcl_msg = parse_point_cloud(pcl_data, "map", jump=30)

    r = rospy.Rate(1)  # 1 Hz로 발행
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
