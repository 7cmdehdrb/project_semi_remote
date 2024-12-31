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

class ManipulatorControl:
    def __init__(self):
        self.tf_listener = tf.TransformListener()
        rospy.sleep(1)
        self.init_pose = self.get_initial_pose()
        self.prev_pose, self.prev_rot, self.prev_time = None, None, None
        self.pub = rospy.Subscriber('/servo_server/target_pose', PoseStamped, queue_size=10)
        rospy.Subscriber('/vacuum_status', Int32MultiArray, self.vacuum_status_callback)
        rospy.Subscriber('/tf_B2P_selected', Float32MultiArray, self.run)
        rospy.Subscriber('/tf_B2P_selected', Float32MultiArray, self.tf_B2P_selected_callback)
        self.target_reached_pub = rospy.Publisher('/target_reached', Bool, queue_size=10)
        self.homepose_reached_pub = rospy.Publisher('/homepose_reached', Bool, queue_size=10)
        self.target_bool = False
        self.vacuum_bool = False
        self.target_pose = geometry_msgs.msg.Pose()
        self.target_position = []
        self.target_orientation = []
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander('manipulator')
        self.tf_B2P_selected_msg = None
        self.placement_complete = False

    def vacuum_status_callback(self, msg):
        if msg.data[0] >= 200 and msg.data[1] >= 200:
            self.vacuum_bool = True
        else:
            self.vacuum_bool = False
        
    def get_initial_pose(self):
        try:
            trans, rot = self.tf_listener.lookupTransform('/base_link', '/tool0', rospy.Time(0))
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = trans
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = rot
            return pose
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr("Error in get_initial_pose: %s", e)
            return None
        
    def calculate_speed(self, prev_pose, current_pose, prev_time, current_time):
        if prev_pose is None or current_pose is None:
            return None
        
        delta_distance = np.linalg.norm(np.array(current_pose) - np.array(prev_pose))
        delta_time = (current_time - prev_time).to_sec()
        
        if delta_time > 0:
            return delta_distance / delta_time
        else:
            return None
                
    def calculate_angular_speed(self, prev_rot, current_rot, prev_time, current_time):
        if prev_rot is None or current_rot is None:
            return None
            
        # Convert quaternions to Euler angles
        prev_euler = euler_from_quaternion(prev_rot)
        current_euler = euler_from_quaternion(current_rot)
        
        # Calculate angular change in each axis
        delta_angles = np.array(current_euler) - np.array(prev_euler)
        delta_time = (current_time - prev_time).to_sec()

        if delta_time > 0:
                # Compute angular speed (rad/s)
                angular_speed = delta_angles / delta_time
                return angular_speed
        else:
                return None
            
    def is_close(self, current_pose, target_pose, position_threshold=0.05, orientation_threshold=0.05):
        # Calculate the difference in position
        # print("current_pose:", current_pose)
        # print("target_pose:", target_pose)
        pos_diff = np.linalg.norm(np.array([
            current_pose.pose.position.x - target_pose.position.x,
            current_pose.pose.position.y - target_pose.position.y,
            current_pose.pose.position.z - target_pose.position.z
        ]))

        # Calculate the difference in orientation
        current_orientation = [current_pose.pose.orientation.x, current_pose.pose.orientation.y,
                               current_pose.pose.orientation.z, current_pose.pose.orientation.w]
        target_orientation = [target_pose.orientation.x, target_pose.orientation.y,
                              target_pose.orientation.z, target_pose.orientation.w]
        orient_diff = angle_between_quaternions(current_orientation, target_orientation)

        return pos_diff < position_threshold #and orient_diff < orientation_threshold

    def run(self, msg):
        self.target_bool = True
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position.x = msg.data[1] + 0.08
        pose_goal.position.y = msg.data[2] - 0.02
        pose_goal.position.z = msg.data[3] + 0.01
        pose_goal.orientation = self.init_pose.pose.orientation
        
        intermediate_pose = copy.deepcopy(pose_goal)
        intermediate_pose.position.x = msg.data[1] + 0.05
        waypoints = []
        waypoints.append(copy.deepcopy(pose_goal))
        waypoints.append(copy.deepcopy(intermediate_pose))
        
        (plan, fraction) = self.move_group.compute_cartesian_path(waypoints, 0.01, 0.0)
        plan = self.move_group.retime_trajectory(self.move_group.get_current_state(), plan, velocity_scaling_factor = 0.5, acceleration_scaling_factor = 0.5)
        self.move_group.execute(plan, wait = True)

        # self.move_group.set_pose_target(pose_goal)
        # success = self.move_group.go(wait=True)

        # self.target_pose = pose_goal
        self.target_pose = intermediate_pose
        # print("Target pose published.")
        self.placement_complete = False
        
    def tf_B2P_selected_callback(self, msg):
        self.tf_B2P_selected_msg = msg
        
    def place(self):
        waypoints = []
        finalpoint = []
        
        if self.tf_B2P_selected_msg is not None:
            intermediate_pose = geometry_msgs.msg.Pose()
            intermediate_pose.position.x = self.tf_B2P_selected_msg.data[1] + 0.4
            intermediate_pose.position.y = self.tf_B2P_selected_msg.data[2]
            intermediate_pose.position.z = self.tf_B2P_selected_msg.data[3] + 0.03
            intermediate_pose.orientation = self.init_pose.pose.orientation
            waypoints.append(copy.deepcopy(intermediate_pose))
        
        intermediate_pose_2 = geometry_msgs.msg.Pose()
        intermediate_pose_2.position.x = self.tf_B2P_selected_msg.data[1] + 0.4
        intermediate_pose_2.position.y = self.tf_B2P_selected_msg.data[2] + 0.4
        intermediate_pose_2.position.z = self.tf_B2P_selected_msg.data[3] + 0.03
        intermediate_pose_2.orientation = self.init_pose.pose.orientation
        waypoints.append(copy.deepcopy(intermediate_pose_2))
        
        plan, fraction = self.move_group.compute_cartesian_path(waypoints, 0.01, 0.0)
        plan = self.move_group.retime_trajectory(self.move_group.get_current_state(), plan, velocity_scaling_factor = 0.5, acceleration_scaling_factor = 0.5)
        self.move_group.execute(plan, wait = True)
        
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation = self.init_pose.pose.orientation
        pose_goal.position.x = -0.70
        pose_goal.position.y = 0.40
        pose_goal.position.z = 0.21
        # angle1 = -math.pi/2
        # quaternion = tf.transformations.quaternion_from_euler(angle1, 0, 0)
        # pose_goal.orientation.x = quaternion[0]
        # pose_goal.orientation.y = quaternion[1]
        # pose_goal.orientation.z = quaternion[2]
        # pose_goal.orientation.w = quaternion[3]
        finalpoint.append(copy.deepcopy(pose_goal))
        plan, fraction = self.move_group.compute_cartesian_path(finalpoint, 0.01, 0.0)
        plan = self.move_group.retime_trajectory(self.move_group.get_current_state(), plan, velocity_scaling_factor = 0.02, acceleration_scaling_factor = 0.02)
        self.move_group.execute(plan, wait = True)
        self.target_reached_pub.publish(Bool(False))
        self.placement_complete = True
        self.go_to_home_pose()
        
    def get_current_orientation(self):
        try:
            trans, rot = self.tf_listener.lookupTransform('/base_link', '/tool0', rospy.Time(0))
            return rot
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr("Error in get_current_orientation: %s", e)
            return None
        
    def go_to_home_pose(self):
        
        home_position = [-0.370, 0.133, 0.142]
        home_quaternion = [-0.500, -0.500, 0.500, 0.500]
        # home_position = [0.181, 0.358, 0.343]
        # home_quaternion = [0.699, 0.048, -0.044, -0.713]
        current_orientation = self.get_current_orientation()
        if current_orientation is None:
            rospy.logerr("Error in go_to_home_pose: current_orientation is None")
            return
        
        home_pose = geometry_msgs.msg.Pose()
        home_pose.position.x = home_position[0]
        home_pose.position.y = home_position[1]
        home_pose.position.z = home_position[2]
        # home_pose.orientation = self.init_pose.pose.orientation
        home_pose.orientation.x = home_quaternion[0]
        home_pose.orientation.y = home_quaternion[1]
        home_pose.orientation.z = home_quaternion[2]
        home_pose.orientation.w = home_quaternion[3]
        
        plan, fraction = self.move_group.compute_cartesian_path([home_pose], 0.01, 0.0)
        plan = self.move_group.retime_trajectory(self.move_group.get_current_state(), plan, velocity_scaling_factor = 0.05, acceleration_scaling_factor = 0.05)
        self.move_group.execute(plan, wait = True)
        self.homepose_reached_pub.publish(Bool(True))
        
    def run_roslaunch_command(self):
        try:
            command = "roslaunch pcd_publisher publish_pcd.launch"
            subprocess.run(command.split())
            rospy.loginfo("roslaunch command executed.")
        except Exception as e:
            rospy.logerr("Error in run_roslaunch_command: %s", e)        

    def while_running(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():        
            # print("target pose:", self.target_bool, "/ vacuum status:", self.vacuum_bool)
            if self.target_bool:
                if self.is_close(self.get_initial_pose(), self.target_pose):
                    rospy.loginfo("Target reached!")
                    self.target_reached_pub.publish(Bool(True))
                    self.target_bool = False
                    
            elif self.vacuum_bool and not self.placement_complete:
                self.place()
                # self.target_reached_pub.publish(Bool(False))
            rate.sleep()
                
def angle_between_quaternions(q1, q2):
    # Function to calculate the angle between two quaternions
    dot_product = np.dot(q1, q2)
    return 2 * math.acos(min(1.0, max(-1.0, dot_product)))

if __name__ == '__main__':
    rospy.init_node('start_pickandplace')
    manipulator_control = ManipulatorControl()
    manipulator_control.while_running()
    rospy.spin()