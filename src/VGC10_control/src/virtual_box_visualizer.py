#!/usr/bin/env python3
import rospy
import os
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, Vector3
from std_msgs.msg import ColorRGBA, Int32MultiArray

def create_box_marker(offset_x, offset_y, offset_z, action):
    marker = Marker()
    marker.header.frame_id = "tool0_controller"  # Reference frame
    marker.header.stamp = rospy.Time.now()
    marker.ns = "virtual_box"
    marker.id = 0
    marker.type = Marker.MESH_RESOURCE
    marker.action = action

    # Offset position
    marker.pose = Pose()
    marker.pose.position.x = offset_x
    marker.pose.position.y = offset_y
    marker.pose.position.z = offset_z
    marker.pose.orientation.w = 1.0
    
    stl_file_path = "file://" + os.path.join(os.environ['HOME'], "vslam_ws/src/collision_detect/description/meshes/virtual_box.stl")
    marker.mesh_resource = stl_file_path

    # Size and color of the box
    marker.scale = Vector3(0.001, 0.001, 0.001)
    marker.color = ColorRGBA(1.0, 0.9, 0.6, 1.0)  # Pinkish color

    return marker

def vacuum_status_callback(msg):
    global box_visible
    # print(msg.data[0], msg.data[1])
    if msg.data[0] >= 200 and msg.data[1] >= 200:
        box_visible = True
    else:
        box_visible = False

    # rospy.Timer(rospy.Duration(10), turn_off_box, oneshot=True)
    
def turn_off_box(event):
    global box_visible
    box_visible = False

def main():
    global box_visible
    
    rospy.init_node('virtual_box_visualizer', anonymous=True)
    rospy.Subscriber('/vacuum_status', Int32MultiArray, vacuum_status_callback)

    marker_pub = rospy.Publisher('virtual/box', Marker, queue_size=10)
    rate = rospy.Rate(120)

    offset_x = 0.0
    offset_y = 0.0
    offset_z = -0.03

    box_visible = True

    while not rospy.is_shutdown():

        if box_visible:
            box_marker = create_box_marker(offset_x, offset_y, offset_z, Marker.ADD)
            marker_pub.publish(box_marker)
        else:
            box_marker = create_box_marker(offset_x, offset_y, offset_z, Marker.DELETE)
            marker_pub.publish(box_marker)

        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
