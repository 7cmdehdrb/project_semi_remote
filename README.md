# Project: Semi-Autonomous Telemanipulation Order Picking Control

This repository implements a semi-remote robotic manipulation system leveraging ROS and Unity. The project includes multiple packages for target object recognition, operator intent estimation, and mobile manipulator control, specifically designed for the UR5e robot with integrated peripherals.

---

## Repository Structure

### 1. `base_package`
This package provides reusable class definitions and header files that can be instantiated by other ROS nodes. It is designed to avoid running separate nodes for common functionalities.

[Link to `base_package`](./src/base_package)

---

### 2. `collision_detect`
This package includes the dependencies and configurations necessary for attaching a custom-designed mobile base to the UR5e manipulator using MoveIt for collision detection.

[Link to `collision_detect`](./src/collision_detect)

---

### 3. `custom_msgs`
This package defines the custom ROS messages used across the project. These messages enable efficient communication between ROS nodes.

[Link to `custom_msgs`](./src/custom_msgs)

---

### 4. `intention`
This is the core package of the project and contains:
- Vision-based target object recognition
- Intersection-based operator intent estimation
- Manual and Autonomous Input controllers

[Link to `intention`](./src/intention)

---

### 5. `log_file`
A folder for storing data used in analysis and debugging.

[Link to `log_file`](./src/log_file)

---

### 6. `moveit_packages`
MoveIt configurations and setup for the UR5e manipulator to enable motion planning and execution.

[Link to `moveit_packages`](./src/moveit_packages)

---

### 7. `realsense-ros`
Manufacturer-provided library and drivers for Intel RealSense D405 camera integration.

[Link to `realsense-ros`](./src/realsense-ros)

---

### 8. `ROS-TCP-Endpoint`
A third-party package enabling TCP/IP communication between Unity-based clients and ROS.

[Link to `ROS-TCP-Endpoint`](./src/ROS-TCP-Endpoint)

---

### 9. `Universal_Robots_ROS_Driver`
Manufacturer-provided driver for interfacing with UR robots in ROS.

[Link to `Universal_Robots_ROS_Driver`](./src/Universal_Robots_ROS_Driver)

---

### 10. `VGC10_control`
Manufacturer-provided library for controlling the VGC10 suction gripper.

[Link to `VGC10_control`](./src/VGC10_control)

---

## How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/7cmdehdrb/project_semi_remote.git
   cd project_semi_remote
   ```

2. Build the project and set environment variable
    ```bash
    cd /home/workspace
    rosdep install --from-paths src --ignore-src -r -y
    catkin_make
    source devel/setup.bash
    ```

3. Run nodes
    ```bash
    roscore
    roslaunch intention launch_base.launch
    roslaunch launch_intention.launch
    ```

---

## <i>Important Notice</i>

**This repository does not include the client program for Meta Quest2.**  
You must set up a separate client to publish the following messages:

- `/controller/right/joy` (`/sensor_msgs/Joy`)
- `/controller/right/twist` (`/geometry_msgs/Twist`)

Additionally, ensure that the IP addresses for the manipulator (RTDE communication) and the suction gripper are configured separately.

---

## Analysis Results

### CSV Data
The file [final_results_by_params_2.csv](./src/log_file/final_results_by_params_2.csv) contains key analysis data. Below is a preview of the file's content:


| Bayesian Threshold | Intersection Length Threshold | Avg Success Time Rate | Max Success Time Rate | Min Success Time Rate | Avg Success Rate |
|---------------------|-------------------------------|------------------------|------------------------|------------------------|------------------|
| 0.1                 | 1                             | 0.3241                 | 0.7753                 | 0.0208                 | 0.8519           |
| 0.1                 | 5                             | 0.3294                 | 0.7753                 | 0.0353                 | 0.8889           |
| 0.1                 | 9                             | 0.3426                 | 0.7753                 | 0.0824                 | 1.0000           |
| ...       | ...   | ...       | ...   | ...   | ...   |


### Visualization
Based on the analysis, the following results were generated:

![Fig11(0)](./src/log_file/images/Fig11(0).png)

![Fig11(1)](./src/log_file/images/Fig11(1).png)

These images summarize the outcomes of the parameterized evaluations visually.