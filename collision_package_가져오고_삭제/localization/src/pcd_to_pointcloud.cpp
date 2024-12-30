#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/conversions.h>  // pcl_ros 헤더를 포함해야 함
#include <std_msgs/Header.h>
#include <pcl_ros/point_cloud.h>  // 이 헤더를 추가합니다.


int main(int argc, char** argv)
{
    // ROS 초기화
    ros::init(argc, argv, "pcd_to_pointcloud_node");
    ros::NodeHandle nh;

    // PCD 파일 경로
    std::string pcd_file = "/home/workspace/src/localization/resources/241108.pcd";

    // PointCloud2 발행을 위한 퍼블리셔 생성
    ros::Publisher cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/map", 1);

    // PCL PointCloud 객체 생성
    pcl::PointCloud<pcl::PointXYZRGB> cloud;

    // PCD 파일 로드
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(pcd_file, cloud) == -1) {
        ROS_ERROR("Couldn't read the PCD file.");
        return -1;
    }

    ROS_INFO("Successfully loaded PCD file with %zu points.", cloud.size());

    // 다운 샘플링 비율 설정 (예: 10의 배수만 남기기)
    int sampling_rate = 10;
    pcl::PointCloud<pcl::PointXYZRGB> downsampled_cloud;

    for (size_t i = 0; i < cloud.size(); ++i) {
        if (i % sampling_rate == 0) {
            downsampled_cloud.push_back(cloud[i]);
        }
    }

    ROS_INFO("Downsampled cloud has %zu points.", downsampled_cloud.size());

    // PointCloud2 메시지 생성
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(downsampled_cloud, cloud_msg);

    // 헤더 정보 설정
    cloud_msg.header.stamp = ros::Time::now();
    cloud_msg.header.frame_id = "map";  // 원하는 프레임 ID 설정

    // 메시지 발행 주기 설정
    ros::Rate loop_rate(1);  // 1Hz로 발행

    int i = 0;

    while (ros::ok()) {
        if (i < 10) {
            cloud_pub.publish(cloud_msg);
            ROS_INFO("Published PointCloud2 message %d times.", i);
        }
        
        ros::spinOnce();

        i++;
        loop_rate.sleep();
    }

    return 0;
}
