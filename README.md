# Quanser ACC 2026 Submission

This repository contains the autonomy stack for the Quanser QCar 2 platform, implemented using ROS 2. It provides a comprehensive suite of tools for state estimation, SLAM, perception, and path following.

## Overview

The stack is designed to enable autonomous navigation for the QCar 2. Key components include:

- **Perception:** Real-time object detection using YOLOv11 (traffic lights, signs, etc.).
- **SLAM & Localization:** SLAM Toolbox integration for mapping and localization, augmented with EKF and IMU-based odometry.
- **Path Planning & Following:** Various controllers including Stanley, Pure Pursuit, MPC, along with a roadmap-based path publisher.
- **Hardware Integration:** ROS 2 nodes for interfacing with QCar 2 sensors and actuators.

## Technical Implementation Details

### 1. Perception Stack
- **Model:** YOLOv11 (trained on custom dataset, weight file: `best2.pt`).
- **Input:** CSI camera streams from the QCar 2.
- **Functionality:** Detects traffic light states and road signs to provide flags for the trip planner.

### 2. State Estimation & Localization
- **IMU Odometry:** `imu_odom` node processes raw IMU and wheel encoder data to provide high-frequency odometry.
- **EKF (Extended Kalman Filter):** A C++ based EKF node (`qcar2_cpp`) fuses IMU/Odom with SLAM pose data for robust state estimation.
- **SLAM:** Uses `slam_toolbox` in asynchronous mode for mapping and localization.

### 3. Control Algorithms
- **Stanley Controller:**
  - Parameters: Wheelbase (0.257m), Gain (k_path: 0.45), Max Speed (0.3 m/s).
  - Uses cross-track error and heading error for steering control.
- **Model Predictive Control (MPC):**
  - Horizon (T): 10 steps.
  - State Vector (NX): [x, y, v, yaw].
  - Solver: `cvxpy` with a linearized vehicle model.
  - Optimization: Minimizes state error and control effort (acceleration and steering).

### 4. Navigation Architecture
- **Roadmap:** Uses the `SDCSRoadMap` utility to generate paths from node sequences.
- **Twist Mux:** A `twist_mux` node manages command priorities (e.g., manual override vs. autonomous navigation).

## Repository Structure

- `ros2/src/`: Core ROS 2 packages.
  - `qcar2_autonomy`: Python-based autonomy nodes (Stanley, YOLO, MPC, etc.).
  - `qcar2_nodes`: Configuration and launch files for platform-level nodes and SLAM.
  - `qcar2_interfaces`: Custom message definitions.
  - `qcar2_cpp`: C++ implementations of performance-critical nodes like EKF.
- `python_resources/`: Standalone scripts, hardware tests, and mapping utilities provided by Quanser.
- `Parameters/`: Configuration files for SLAM and twist multiplexing.
- `rviz/`: RViz2 configuration files for visualization.

## Instructions

Follow the steps below to build and launch the project:

1. **Clone this repository** into your workspace (e.g., `ACC_DEVELOPMENT` folder).
2. **Navigate to the ROS 2 workspace**:
   
   ```bash
   cd ros2
   colcon build --symlink-install
   source install/setup.bash
   ros2 launch qcar2_autonomy my_autonomy_launch.py
   ```

3. **Launch Details**:
   - The `my_autonomy_launch.py` script starts the RViz2 visualization, QCar 2 platform nodes, SLAM localization, state estimation (EKF/IMU), and the Stanley controller.
   - It also initiates the YOLO detector for object recognition.

## Dependencies

### Python Dependencies
Ensure the following Python packages are installed:
- `ultralytics`, `cvxpy`, `scipy`


See the video demonstration [link_1]()     [link_2](https://youtu.be/qH_KObXcGQ8?si=vGFEC_wz_y0v0Vxl)


