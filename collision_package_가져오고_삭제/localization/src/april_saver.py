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

from april_tag_localization import AprilTagDetector
import json


class Saver:
    def __init__(self):
        self.april_detector = AprilTagDetector()
        self.trigger_sub = rospy.Subscriber("/trigger", Empty, self.trigger_callback)

    def save(self, file_path):
        json_data = self.april_detector.tags.to_json()

        with open(file_path, "w") as f:
            # json.dump()를 사용하여 딕셔너리 데이터를 JSON 파일로 저장
            json.dump(json_data, f, indent=4)

    def trigger_callback(self, msg: Bool):
        self.save(file_path="/home/irol/project_hj/src/localization/resources/1205_april_data.json")

        self.trigger_sub.unregister()


def main():
    rospy.init_node("april_saver_node")  # TODO: Add node name

    saver = Saver()

    # rospy.spin()

    r = rospy.Rate(5)  # TODO: Add rate
    while not rospy.is_shutdown():

        print(saver.april_detector.tags.to_json())

        r.sleep()


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
