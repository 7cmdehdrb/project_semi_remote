#include <ros/ros.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit_msgs/CollisionObject.h>
#include <geometric_shapes/shapes.h>
#include <geometric_shapes/shape_operations.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Vector3.h>
#include <std_msgs/Bool.h>
#include <std_msgs/ColorRGBA.h>
#include <std_msgs/Int32MultiArray.h>
#include <sensor_msgs/JointState.h>
#include <visualization_msgs/Marker.h>

bool box_visible = true;

visualization_msgs::Marker createBoxMarker(float offset_x, float offset_y, float offset_z, uint32_t action) {
    visualization_msgs::Marker vb_marker;

    vb_marker.header.frame_id = "tool0";
    vb_marker.header.stamp = ros::Time::now();
    vb_marker.ns = "virtual_box";
    vb_marker.id = 0;
    vb_marker.type = visualization_msgs::Marker::MESH_RESOURCE;
    vb_marker.action = action;

    // Offset position
    vb_marker.pose.position.x = offset_x;
    vb_marker.pose.position.y = offset_y;
    vb_marker.pose.position.z = offset_z;
    vb_marker.pose.orientation.w = 1.0;

    std::string stl_file_path = "file:///home/irol/vslam_ws/src/collision_detect/description/meshes/virtual_box.stl";
    vb_marker.mesh_resource = stl_file_path;

    // Size and color of the box
    vb_marker.scale.x = 0.001;
    vb_marker.scale.y = 0.001;
    vb_marker.scale.z = 0.001;
    vb_marker.color.r = 1.0;
    vb_marker.color.g = 0.9;
    vb_marker.color.b = 0.6;
    vb_marker.color.a = 0.8; // Don't forget to set the alpha!

    return vb_marker;
}

void vacuumStatusCallback(const std_msgs::Int32MultiArray::ConstPtr& msg) {
    if (msg->data[0] >= 200 && msg->data[1] >= 200) {
        box_visible = true;
    } else {
        box_visible = false;
    }
}

void jointStateCallback(const sensor_msgs::JointState::ConstPtr& joint_state, robot_state::RobotState* current_state)
{
    current_state->setVariableValues(*joint_state);
    current_state->update();
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "fov_collision_detector");
    ros::NodeHandle nh;
    ros::AsyncSpinner spinner(1);
    spinner.start();

    // Load the robot model
    robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
    robot_model::RobotModelPtr robot_model = robot_model_loader.getModel();

    // Create a planning scene
    planning_scene::PlanningScene planning_scene(robot_model);

    // Load the URDF model of the camera fov
    shapes::Mesh* m = shapes::createMeshFromResource("file:///home/irol/vslam_ws/src/collision_detect/description/meshes/FOV.stl");
    shape_msgs::Mesh mesh;
    shapes::ShapeMsg mesh_msg;
    shapes::constructMsgFromShape(m, mesh_msg);
    mesh = boost::get<shape_msgs::Mesh>(mesh_msg);
    
    moveit_msgs::CollisionObject collision_object;
    collision_object.header.frame_id = "base_link";
    collision_object.id = "fov";

    geometry_msgs::Pose fov_pose;
    fov_pose.position.x = -0.35;
    fov_pose.position.y = -0.36;
    fov_pose.position.z = 0.55;
    fov_pose.orientation.x = -0.5;
    fov_pose.orientation.y = -0.5;
    fov_pose.orientation.z = 0.5;
    fov_pose.orientation.w = 0.5;

    collision_object.meshes.push_back(mesh);
    // collision_object.mesh_poses.push_back(geometry_msgs::Pose());
    collision_object.mesh_poses.push_back(fov_pose);
    collision_object.operation = collision_object.ADD;

    // Add the collision object into the planning scene
    planning_scene.processCollisionObjectMsg(collision_object);

    // Set up the marker for FOV
    visualization_msgs::Marker fov_marker;
    fov_marker.header.frame_id = "base_link";
    fov_marker.header.stamp = ros::Time::now();
    fov_marker.ns = "fov";
    fov_marker.id = 0;
    fov_marker.type = visualization_msgs::Marker::MESH_RESOURCE;
    fov_marker.action = visualization_msgs::Marker::ADD;
    fov_marker.pose = fov_pose;
    fov_marker.scale.x = 0.001;
    fov_marker.scale.y = 0.001;
    fov_marker.scale.z = 0.001;
    fov_marker.color.r = 0.0;
    fov_marker.color.g = 0.7;
    fov_marker.color.b = 0.0;
    fov_marker.color.a = 0.5; // Semi-transparent

    // Set the mesh resource for the marker
    fov_marker.mesh_resource = "file:///home/irol/vslam_ws/src/collision_detect/description/meshes/FOV.stl";

    // Get the current state of the robot
    robot_state::RobotState& current_state = planning_scene.getCurrentStateNonConst();
    ros::Subscriber joint_state_sub = nh.subscribe<sensor_msgs::JointState>("/joint_states", 10, boost::bind(jointStateCallback, _1, &current_state));
    ros::Subscriber vacuum_sub = nh.subscribe("/vacuum_status", 10, vacuumStatusCallback);

    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 1);
    ros::Publisher collision_pub = nh.advertise<std_msgs::Bool>("collision", 1);
    ros::Publisher box_pub = nh.advertise<visualization_msgs::Marker>("virtual/box", 10);

    ros::Rate loop_rate(120);

    float offset_x = 0.0;
    float offset_y = 0.0;
    float offset_z = 0.15;

    while (ros::ok())
    {
        collision_detection::CollisionRequest collision_request;
        collision_detection::CollisionResult collision_result;

        planning_scene.setCurrentState(current_state);
        planning_scene.checkCollision(collision_request, collision_result, current_state);

        if (collision_result.collision)
        {
            ROS_INFO("Collision detected");
        }
        else
        {
            ROS_INFO("No collision");
        }

        std_msgs::Bool collision_msg;
        collision_msg.data = collision_result.collision;
        collision_pub.publish(collision_msg);
        marker_pub.publish(fov_marker);


        if (box_visible) {
            visualization_msgs::Marker box_marker = createBoxMarker(offset_x, offset_y, offset_z, visualization_msgs::Marker::ADD);
            box_pub.publish(box_marker);
        } else {
            visualization_msgs::Marker box_marker = createBoxMarker(offset_x, offset_y, offset_z, visualization_msgs::Marker::DELETE);
            box_pub.publish(box_marker);
        }

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}