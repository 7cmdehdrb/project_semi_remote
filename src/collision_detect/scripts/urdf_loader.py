#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from shape_msgs.msg import Mesh, MeshTriangle
import numpy as np
from stl import mesh

def load_mesh_from_stl(file_path):
    # Function to load an STL file and convert it to a Mesh message

    your_mesh = mesh.Mesh.from_file(file_path)
    mesh_msg = Mesh()

    for i in range(len(your_mesh.vectors)):
        triangle = MeshTriangle()
        triangle.vertex_indices = [i * 3, i * 3 + 1, i * 3 + 2]
        mesh_msg.triangles.append(triangle)

        for vector in your_mesh.vectors[i]:
            point = geometry_msgs.msg.Point()
            point.x, point.y, point.z = vector
            mesh_msg.vertices.append(point)

    return mesh_msg

def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('collision_detector', anonymous=True)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()

    # Load FOV mesh
    fov_mesh_path = "/home/irol/vslam_ws/src/collision_detect/description/meshes/FOV.stl"
    fov_mesh = load_mesh_from_stl(fov_mesh_path)

    # Add FOV as a collision object
    fov_collision_object = moveit_msgs.msg.CollisionObject()
    fov_collision_object.id = "fov"
    fov_collision_object.meshes = [fov_mesh]
    fov_collision_object.mesh_poses = [geometry_msgs.msg.Pose()]
    fov_collision_object.operation = fov_collision_object.ADD

    scene.add_collision_object(fov_collision_object)

    rospy.sleep(2)  # Wait for the scene to update

    # Function to check for collision
    def check_collision():
        current_state = robot.get_current_state()
        for collision_object in scene.get_known_object_names():
            if scene.is_collision_object_attached(collision_object):
                continue
            if robot.check_collisions_with_collision_object(current_state, collision_object):
                print(f"Collision detected with {collision_object}")
                return True
        print("No collision detected")
        return False

    # Test Collision Detection
    check_collision()

if __name__ == '__main__':
    main()