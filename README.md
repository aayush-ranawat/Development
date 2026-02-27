## Instructions

Follow the steps below to build and launch the project:

1. Clone this repository into the `ACC_DEVELOPMENT` folder.
2. Navigate to the ROS 2 workspace:
   
   ```bash
   cd ros2
   colcon build --symlink-install
   ros2 launch qcar2_autonomy my_autonomy_launch.py

Additional Dependencies

Make sure the following Python packages are installed:

ultralytics

cvxpy (required only if using the MPC node)

You can install them by adding command in dockerfile or using:
   ```bash
pip install ultralytics cvxpy



