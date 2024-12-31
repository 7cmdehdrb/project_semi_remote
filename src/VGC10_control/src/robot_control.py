#!/usr/bin/env python3
import roslib
import rospy
import math
import tf
import geometry_msgs.msg
from geometry_msgs.msg import PoseStamped
import sensor_msgs.msg
from std_msgs.msg import Float32MultiArray, Bool, Int32MultiArray
import numpy as np
from tf.transformations import *
import moveit_commander
import copy
import moveit_msgs.msg
import subprocess



rospy.init_node('robot_control')

listener = tf.TransformListener()
# pose_publisher = rospy.Publisher('tf_B2T', Float32MultiArray, queue_size=10)
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
move_group = moveit_commander.MoveGroupCommander('manipulator')

rate = rospy.Rate(10.0)
i = 0

while not rospy.is_shutdown():
    try:
        if i == 0:
            (trans, rot) = listener.lookupTransform('base_link', 'tool0', rospy.Time(0))
            # pose_goal = geometry_msgs.msg.Pose()
            # pose_goal.position.x = trans[0] - 0.3
            # pose_goal.position.y = trans[1]
            # pose_goal.position.z = trans[2]
            # pose_goal.orientation.x = rot[0]
            # pose_goal.orientation.y = rot[1]
            # pose_goal.orientation.z = rot[2]
            # pose_goal.orientation.w = rot[3]
            # move_group.set_pose_target(pose_goal)

            # plan = move_group.go(wait=False)
            pose_goal = geometry_msgs.msg.Pose()
            pose_goal.position.x = trans[0] - 0.4
            pose_goal.position.y = trans[1]
            pose_goal.position.z = trans[2]
            pose_goal.orientation.x = rot[0]
            pose_goal.orientation.y = rot[1]
            pose_goal.orientation.z = rot[2]
            pose_goal.orientation.w = rot[3]
            waypoints = []
            waypoints.append(copy.deepcopy(pose_goal))
            # waypoints.append(copy.deepcopy(intermediate_pose))

            (plan, fraction) = move_group.compute_cartesian_path(waypoints, 0.001, 0.0)
            plan = move_group.retime_trajectory(move_group.get_current_state(), plan, velocity_scaling_factor = 0.5, acceleration_scaling_factor = 0.5)
            # move_group.setMaxVelocityScalingFactor(1)
            # move_group.set_max_velocity_scaling_factor(1)
            move_group.execute(plan, wait = True)
            i += 1
        
        
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        continue
    
    rate.sleep()