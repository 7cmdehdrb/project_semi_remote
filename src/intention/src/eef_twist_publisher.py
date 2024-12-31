#!/usr/bin/python3

import rospy
import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, Vector3
from moveit_commander import RobotCommander, MoveGroupCommander


class JacobianTwistCalculator:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("jacobian_twist_calculator")

        # Subscribers
        self.joint_states_subscriber = rospy.Subscriber(
            "/joint_states", JointState, self.joint_states_callback
        )

        # Publisher
        self.twist_publisher = rospy.Publisher(
            "/end_effector/twist", Twist, queue_size=10
        )

        # MoveIt Commander
        self.robot_commander = RobotCommander()
        self.move_commander = MoveGroupCommander("manipulator")

        # Joint velocities placeholder
        self.joint_velocities = [0.0] * 6

    def joint_states_callback(self, msg):
        self.joint_velocities = msg.velocity

    def calculate_jacobian_twist(self):
        # Get the Jacobian matrix
        J = self.move_commander.get_jacobian_matrix(
            self.move_commander.get_current_joint_values()
        )

        # Change joint velocity index as required
        def change_idx(joint_velocities):
            new_joint_velocities = [0.0] * 6
            new_joint_velocities[0] = joint_velocities[2]
            new_joint_velocities[1] = joint_velocities[1]
            new_joint_velocities[2] = joint_velocities[0]
            new_joint_velocities[3] = joint_velocities[3]
            new_joint_velocities[4] = joint_velocities[4]
            new_joint_velocities[5] = joint_velocities[5]
            return new_joint_velocities

        # Apply index change
        joint_velocities = np.array(change_idx(self.joint_velocities))

        # Calculate linear and angular velocities
        linear_velocity = np.dot(J, joint_velocities)

        # Populate Twist message
        data = Twist()
        data.linear = Vector3(
            x=linear_velocity[0], y=linear_velocity[1], z=linear_velocity[2]
        )
        data.angular = Vector3(
            x=linear_velocity[3], y=linear_velocity[4], z=linear_velocity[5]
        )

        return data

    def run(self):
        rate = rospy.Rate(30)  # 10 Hz
        while not rospy.is_shutdown():
            # Calculate twist
            twist = self.calculate_jacobian_twist()

            # Publish twist
            self.twist_publisher.publish(twist)

            # rospy.loginfo(f"Published Twist: {twist}")

            rate.sleep()


if __name__ == "__main__":
    try:
        calculator = JacobianTwistCalculator()
        calculator.run()
    except rospy.ROSInterruptException:
        pass
