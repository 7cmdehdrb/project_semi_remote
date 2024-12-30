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
from interactive_markers.interactive_marker_server import *


# MoveIt
from moveit_commander import RobotCommander, PlanningSceneInterface, MoveGroupCommander
from moveit_msgs.srv import (
    GetStateValidity,
    GetStateValidityRequest,
    GetStateValidityResponse,
)


quat = Quaternion(x=-0.5, y=0.5, z=-0.5, w=0.5)


class InteractiveMarkerController(object):
    def __init__(self):
        self.server = InteractiveMarkerServer("interactive_markers")

        self.create_interactive_marker(
            name="area1", position=[0.2, 0.0, 0.0], color=[1.0, 0.0, 0.0]
        )
        # self.create_interactive_marker(
        #     name="area2", position=[0.4, 0.0, 0.0], color=[0.0, 1.0, 0.0]
        # )
        # self.create_interactive_marker(name="area3", position=[0.6, 0.0, 0.2], color=[0.0, 0.0, 1.0])

        self.server.applyChanges()

        print("Interactive Marker Controller Initialized.")

    def create_interactive_marker(self, name, position, color):
        # Interactive Marker 생성
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "map"
        int_marker.name = name
        int_marker.description = f"Marker: {name}"
        int_marker.scale = 0.5

        # 초기 위치 설정
        int_marker.pose.position.x = position[0]
        int_marker.pose.position.y = position[1]
        int_marker.pose.position.z = position[2]

        # Marker 생성
        box_marker = Marker()
        box_marker.type = Marker.SPHERE
        box_marker.scale.x = 0.1
        box_marker.scale.y = 0.1
        box_marker.scale.z = 0.1
        box_marker.color.r = color[0]
        box_marker.color.g = color[1]
        box_marker.color.b = color[2]
        box_marker.color.a = 1.0

        # Marker 컨트롤 추가
        box_control = InteractiveMarkerControl()
        box_control.always_visible = True
        box_control.markers.append(box_marker)
        int_marker.controls.append(box_control)

        # x축 이동 컨트롤 추가
        control_x = InteractiveMarkerControl()
        control_x.orientation.w = 1
        control_x.orientation.x = 1
        control_x.orientation.y = 0
        control_x.orientation.z = 0
        control_x.name = f"{name}_move_x"
        control_x.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control_x)

        # y축 이동 컨트롤 추가
        control_y = InteractiveMarkerControl()
        control_y.orientation.w = 1
        control_y.orientation.x = 0
        control_y.orientation.y = 1
        control_y.orientation.z = 0
        control_y.name = f"{name}_move_y"
        control_y.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control_y)

        # z축 이동 컨트롤 추가
        control_z = InteractiveMarkerControl()
        control_z.orientation.w = 1
        control_z.orientation.x = 0
        control_z.orientation.y = 0
        control_z.orientation.z = 1
        control_z.name = f"{name}_move_z"
        control_z.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control_z)

        # Marker를 Interactive Marker Server에 추가
        self.server.insert(int_marker, self.process_feedback)

    def process_feedback(self, feedback):
        pass

    def get_marker_pose(self, name):
        return self.server.get(name).pose


def add_random_pose(pose: Pose):
    pose.position.x += np.random.uniform(-0.1, 0.1)
    pose.position.y += np.random.uniform(-0.1, 0.1)
    pose.position.z += np.random.uniform(-0.1, 0.1)

    pose.orientation = Quaternion(x=-0.5, y=0.5, z=-0.5, w=0.5)
    return pose


class SimulationPlanner:
    def __init__(self):
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group = MoveGroupCommander("ENDEFFECTOR")

        self.marker_server = InteractiveMarkerController()

        self.trigger_sub = rospy.Subscriber("/trigger", Empty, self.trigger_callback)
        self.test_pub = rospy.Publisher(
            "/real_intention_traj/generate_path", Empty, queue_size=1
        )
        # rostopic pub /real_intention_traj/generate_path std_msgs/Empty "{}"
        self.trigger = False

    def trigger_callback(self, msg):
        self.trigger = True
        # self.planning()
        # self.trigger_sub.unregister()

    def homing(self):
        self.group.set_pose_target(
            Pose(
                position=Point(
                    x=0.4474461400315329, y=-0.08294117208740881, z=0.10095998769939118
                ),
                orientation=Quaternion(x=-0.5, y=0.5, z=-0.5, w=0.5),
            )
            # 0.5 -0.5 0.5 0.5
        )
        self.group.go(wait=True)

        print("Homing Completed.")

    def homing_with_joint_values(self):
        rospy.loginfo("Homing...")

        target_joint_values = [
            -2.993309480340355,
            -1.965223182987291,
            -2.3028642984164724,
            4.267264906537596,
            -1.4225131535454585,
            -3.141977238426645,
        ]
        self.group.set_joint_value_target(target_joint_values)
        self.group.go(wait=True)

    def planning(self):
        if self.trigger is False:
            return

        waypoints = [
            Pose(
                position=Point(
                    x=0.4474461400315329, y=-0.08294117208740881, z=0.10095998769939118
                ),
                orientation=Quaternion(x=-0.5, y=0.5, z=-0.5, w=0.5),
            ),
            add_random_pose(self.marker_server.get_marker_pose("area1")),
            # add_random_pose(self.marker_server.get_marker_pose("area2")),
            Pose(
                position=Point(
                    x=0.80887537, y=0.3666989111542225, z=0.2163765698336322
                ),
                orientation=Quaternion(x=-0.5, y=0.5, z=-0.5, w=0.5),
            ),
        ]

        print("Planning Started.")

        plan, fraction = self.group.compute_cartesian_path(
            waypoints, eef_step=0.01, avoid_collisions=False
        )

        if fraction == 1.0:
            print("Planning Succeeded.")
            self.group.execute(plan)
        else:
            print("Planning Failed.")
            self.homing_with_joint_values()


def main():
    rospy.init_node("simulaion_planner_node")  # TODO: Add node name

    planner = SimulationPlanner()

    # planner.homing()
    planner.homing_with_joint_values()

    # rospy.spin()

    r = rospy.Rate(10)  # TODO: Add rate
    while not rospy.is_shutdown():
        planner.planning()
        planner.homing_with_joint_values()
        planner.test_pub.publish(Empty())
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
