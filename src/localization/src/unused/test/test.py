#!/usr/bin/python3

import os
import sys
import roslib.exceptions
import roslib.packages
import rospy
import roslib
import numpy as np
from abc import ABC, abstractmethod

# Moveit!
import moveit_commander

# Messages
from tf.transformations import (
    quaternion_multiply,
    quaternion_from_euler,
    euler_from_quaternion,
)
from std_msgs.msg import *
from sensor_msgs.msg import *
from geometry_msgs.msg import *
from nav_msgs.msg import *
from custom_msgs.msg import *


rospy.init_node("test_node")

move_group = moveit_commander.MoveGroupCommander("ENDEFFECTOR")

r = rospy.Rate(1)
while not rospy.is_shutdown():
    st = move_group.get_current_state()
    print(st, end="\n\n")
    r.sleep()
