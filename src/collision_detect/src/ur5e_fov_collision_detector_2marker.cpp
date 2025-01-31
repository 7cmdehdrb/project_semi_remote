#include <ros/ros.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit_msgs/CollisionObject.h>
#include <geometric_shapes/shapes.h>
#include <geometric_shapes/shape_operations.h>
#include <sensor_msgs/JointState.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Bool.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Vector3.h>  
#include <std_msgs/ColorRGBA.h>
#include <std_msgs/Int32MultiArray.h>

bool box_visible = true;

visualization_msgs::Marker createBoxMarker(float offset_x, float offset_y, float offset_z, int action) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "tool0_controller";
    marker.header.stamp = ros::Time::now();
    marker.ns = "virtual_box";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = action;

    // Offset position
    marker.pose.position.x = offset_x;
    marker.pose.position.y = offset_y;
    marker.pose.position.z = offset_z;
    marker.pose.orientation.w = 1.0;

    // Size and color of the box
    marker.scale.x = 0.27;
    marker.scale.y = 0.15;
    marker.scale.z = 0.18;
    marker.color.r = 1.0;
    marker.color.g = 0.9;
    marker.color.b = 0.6;
    marker.color.a = 1.0; // Alpha must be non-zero

    return marker;
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

    // Get the current state of the robot
    robot_state::RobotState& current_state = planning_scene.getCurrentStateNonConst();
    ros::Subscriber joint_state_sub = nh.subscribe<sensor_msgs::JointState>("/joint_states", 10, boost::bind(jointStateCallback, _1, &current_state));
    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 1);
    ros::Publisher collision_pub = nh.advertise<std_msgs::Bool>("collision", 1);

    // Virtual box
    ros::Subscriber sub = nh.subscribe("/vacuum_status", 10, vacuumStatusCallback);
    ros::Publisher box_marker_pub = nh.advertise<visualization_msgs::Marker>("virtual/box", 10);

    float offset_x = 0.0;
    float offset_y = 0.0;
    float offset_z = 0.05;

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

    ros::Rate loop_rate(60);
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
        marker_pub.publish(fov_marker);


        visualization_msgs::Marker box_marker;
        if (box_visible) {
            box_marker = createBoxMarker(offset_x, offset_y, offset_z, visualization_msgs::Marker::ADD);
        } else {
            box_marker = createBoxMarker(offset_x, offset_y, offset_z, visualization_msgs::Marker::DELETE);
        }
        box_marker_pub.publish(box_marker);


        std_msgs::Bool collision_msg;
        collision_msg.data = collision_result.collision;
        collision_pub.publish(collision_msg);
        
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}