#!/usr/bin/env python3

import rospy
import pcl
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pcl2
import std_msgs.msg
import numpy as np


def publish_pcd():
    rospy.init_node("pcd_publisher", anonymous=True)
    pub = rospy.Publisher("/pcd", PointCloud2, queue_size=1)
    rospy.loginfo("Started PCD publisher")

    pcl_cloud = pcl.load(
        "/home/irol/vslam_ws/src/pcd_publisher/pcd_file/global_240516_rejectedLoopclosure.pcd"
    )
    points = np.array(pcl_cloud, dtype=np.float32)[
        :, :3
    ]  # Extracting only x, y, z coordinates

    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "map"
    ros_cloud = pcl2.create_cloud_xyz32(header, points)

    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        header.stamp = rospy.Time.now()
        ros_cloud.header = header
        pub.publish(ros_cloud)
        rate.sleep()


if __name__ == "__main__":
    try:
        publish_pcd()
    except rospy.ROSInterruptException:
        pass
