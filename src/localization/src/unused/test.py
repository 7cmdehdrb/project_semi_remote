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


def callback(msg: PointCloud2):
    rospy.loginfo("Callback called.")
    for field in msg.fields:
        print(field)

    print("\n")


def main():
    rospy.init_node("test_node2")  # TODO: Add node name

    rospy.Subscriber("/map", PointCloud2, callback)  # TODO: Add subscriber

    rospy.spin()

    # r = rospy.Rate()  # TODO: Add rate
    # while not rospy.is_shutdown():
    #     r.sleep()


if __name__ == "__main__":
    try:
        main()
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
