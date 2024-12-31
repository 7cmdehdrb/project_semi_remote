#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Vector3.h>
#include <std_msgs/ColorRGBA.h>
#include <std_msgs/Int32MultiArray.h>

bool box_visible = true;

visualization_msgs::Marker createBoxMarker(float offset_x, float offset_y, float offset_z, uint32_t action) {
    visualization_msgs::Marker marker;

    marker.header.frame_id = "tool0";
    marker.header.stamp = ros::Time::now();
    marker.ns = "virtual_box";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::MESH_RESOURCE;
    marker.action = action;

    // Offset position
    marker.pose.position.x = offset_x;
    marker.pose.position.y = offset_y;
    marker.pose.position.z = offset_z;
    marker.pose.orientation.w = 1.0;

    std::string stl_file_path = "file://" + std::string(getenv("HOME")) + "/vslam_ws/src/collision_detect/description/meshes/virtual_box.stl";
    marker.mesh_resource = stl_file_path;

    // Size and color of the box
    marker.scale.x = 0.001;
    marker.scale.y = 0.001;
    marker.scale.z = 0.001;
    marker.color.r = 1.0;
    marker.color.g = 0.9;
    marker.color.b = 0.6;
    marker.color.a = 0.8; // Don't forget to set the alpha!

    return marker;
}

void vacuumStatusCallback(const std_msgs::Int32MultiArray::ConstPtr& msg) {
    if (msg->data[0] >= 200 && msg->data[1] >= 200) {
        box_visible = true;
    } else {
        box_visible = false;
    }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "virtual_box_visualizer");
    ros::NodeHandle n;

    ros::Subscriber sub = n.subscribe("/vacuum_status", 10, vacuumStatusCallback);
    ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("virtual/box", 10);

    ros::Rate rate(120); // 120 Hz

    float offset_x = 0.0;
    float offset_y = 0.0;
    float offset_z = 0.15;

    while (ros::ok()) {
        if (box_visible) {
            visualization_msgs::Marker box_marker = createBoxMarker(offset_x, offset_y, offset_z, visualization_msgs::Marker::ADD);
            marker_pub.publish(box_marker);
        } else {
            visualization_msgs::Marker box_marker = createBoxMarker(offset_x, offset_y, offset_z, visualization_msgs::Marker::DELETE);
            marker_pub.publish(box_marker);
        }

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}