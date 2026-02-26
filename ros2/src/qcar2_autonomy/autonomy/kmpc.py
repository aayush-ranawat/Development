#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
# from tf_transformations import euler_from_quaternion
from scipy.spatial.transform import Rotation as R
import numpy as np
import math
from hal.products.mats import SDCSRoadMap

import math
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import cvxpy
import numpy as np
from utils.utils import nearest_point, pi_2_pi, quat_2_rpy
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
# Import the MPC class from your file
# Ensure kmpc_planner.py is in the same folder or python path
from utils.kinematic_mpc import KMPCPlanner, mpc_config

class MPCNode(Node):
    def __init__(self):
        super().__init__('mpc_node')

        # --- 1. Initialize Configuration ---
        self.config = mpc_config()
        
        # Adjust MPC parameters if necessary for your robot
        # self.config.MAX_SPEED = 2.0 
        
        # --- 2. Generate/Load Reference Path ---
        # Format required by planner: List of [cx, cy, cyaw, sp]
        # Here we generate a simple figure-8 or circle for testing
        self.waypoints = self.generate_circle_path(radius=5.0, steps=200)
        print(f"shape is {self.waypoints.shape}")
        
        # --- 3. Initialize the MPC Planner ---
        self.planner = KMPCPlanner(waypoints=self.waypoints, config=self.config)

        # --- 4. ROS Interfaces ---
        # Subscriber: Current State from EKF
        self.pose_sub = self.create_subscription(
            Odometry,
            '/ekf_pose',
            self.pose_callback,
            10
        )
        
        # Publisher: Control Commands
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel_nav',
            10
        )

        # Timer: Run control loop at MPC timestep (DTK)
        # Default DTK is 0.1s (10 Hz) changed to 50hz 0.02
        self.timer = self.create_timer(self.config.DTK, self.control_loop)

        # State storage [x, y, delta, v, yaw, yawrate, beta]
        self.current_state = None 

        self.get_logger().info("MPC Controller Node Started. Waiting for EKF pose...")

    def generate_circle_path(self, radius, steps):
        """Generates a circular path for the robot to follow."""
        cx, cy, cyaw, sp = [], [], [], []
        
            # Configuration parameters for example script.
        useSmallMap = False
        leftHandTraffic = False
        nodeSequence = [10,2,4,6,8,10]
        # Create a SDCSRoadMap instance with desired configuration.
        roadmap = SDCSRoadMap(
            leftHandTraffic=leftHandTraffic,
                useSmallMap=useSmallMap
        )
    # Generate the shortest path passing through the given sequence of nodes.
        # - nodeSequence can be a list or tuple of node indicies.
        # - The generated path takes the form of a 2xn numpy array
        path = roadmap.generate_path(nodeSequence=nodeSequence)*0.93
        path=self.rotate_waypoints(path,-9)
        cx,cy=path[0,:],path[1,:]
        
        cyaw,sp=self.calculate_path_attributes(cx,cy)



        # The planner expects a numpy array where:
        # path[0] = x, path[1] = y, path[2] = yaw, path[3] = speed
        return np.array([cx, cy, cyaw, sp])
    

    def calculate_path_attributes(self,cx, cy, target_speed=0.5, max_lat_accel=0.2):
        """
        Calculates heading (yaw) and velocity (speed) based on x, y coordinates.
        
        Args:
            cx: List of x coordinates
            cy: List of y coordinates
            target_speed: Desired max speed (m/s)
            max_lat_accel: Maximum lateral acceleration allowed (determines cornering speed)
            
        Returns:
            cyaw: List of heading angles (radians)
            sp: List of target velocities (m/s)
        """
        
        # Ensure inputs are numpy arrays
        # cx = np.array(cx)
        # cy = np.array(cy)
        
        # --- 1. Calculate Heading (cyaw) ---
        # We use np.gradient to get the direction of the path at every point
        # dx/dt and dy/dt (assuming constant step size for geometry)
        dx = np.gradient(cx)
        dy = np.gradient(cy)
        
        # Compute yaw using arctan2 (handles all quadrants correctly)
        cyaw = np.arctan2(dy, dx)
        
        # Unwrap ensures the angle is continuous (e.g., doesn't jump from 3.14 to -3.14)
        # This helps controllers avoid sudden steering jerks
        cyaw = np.unwrap(cyaw)

        # --- 2. Calculate Velocity Profile (sp) ---
        # To find the right speed, we first need the curvature (kappa).
        # Formula: k = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
        
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Calculate curvature
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
        
        # Handle NaN or Inf resulting from straight lines (zero division)
        curvature = np.nan_to_num(curvature)
        
        # Calculate Max Speed allowed based on curvature
        # v_max = sqrt(a_lat_max / curvature)
        # We add a small epsilon (1e-6) to curvature to avoid division by zero
        sp = np.sqrt(max_lat_accel / (curvature + 1e-6))
        
        # Clip the speed to not exceed the target_speed (the car's physical limit)
        sp = np.minimum(sp, target_speed)

        # --- 3. Terminal Velocity Handling ---
        # The robot should slow down as it approaches the end of the path
        # We linearly ramp down speed for the last 5 points
        if len(sp) > 5:
            for i in range(5):
                sp[-(i+1)] = sp[-(i+1)] * (i / 5.0)

        return cyaw.tolist(), sp.tolist()
    

    def rotate_waypoints(self,xy, theta):
        """
        xy: (N,2) array of [x,y]
        theta: rotation angle in degree
        """
        theta=np.deg2rad(theta)
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        return R @ xy

    def pose_callback(self, msg: Odometry):
        """Updates the vehicle state from EKF Odometry."""
        # 1. Position
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # 2. Orientation (Quaternion -> Euler)
        q = msg.pose.pose.orientation
        quat_list = [q.x, q.y, q.z, q.w]
        (roll, pitch, yaw) = self.euler_from_quaternion(quat_list)

        # 3. Velocity (Linear X)
        v = msg.twist.twist.linear.x
        
        # 4. Yaw Rate (Angular Z)
        yaw_rate = msg.twist.twist.angular.z

        # 5. Steering Angle (delta) & Side Slip (beta)
        # Note: Odometry usually doesn't provide current steering angle.
        # Ideally, subscribe to /joint_states for this. 
        # For now, we assume 0.0 or rely on MPC's internal prediction model.
        delta = 0.0 
        beta = 0.0

        # Update state vector required by KMPCPlanner
        self.current_state = np.array([x, y, delta, v, yaw, yaw_rate, beta])

    def euler_from_quaternion(self,q):
        # q = [x, y, z, w]
        r = R.from_quat(q)
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        return roll, pitch, yaw

    def control_loop(self):
        """Main MPC Loop."""
        if self.current_state is None:
            return

        try:
            # CALL THE MPC PLAN FUNCTION
            # It returns (steering_angle, speed)
            steer, speed = self.planner.plan(self.current_state)

            # Publish Twist Message
            cmd = Twist()
            cmd.linear.x = float(speed)
            cmd.angular.z = float(steer) # Assuming angular.z controls steering
            self.cmd_pub.publish(cmd)

        except Exception as e:
            self.get_logger().error(f"MPC Solver Failed: {e}")
            # Safety stop
            self.cmd_pub.publish(Twist())

    

def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()