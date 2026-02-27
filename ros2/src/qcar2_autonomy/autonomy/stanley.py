#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import numpy as np
import math
from scipy.spatial.transform import Rotation as Rot
from scipy.interpolate import CubicSpline

# Assuming this is your custom/proprietary map module
from hal.products.mats import SDCSRoadMap 


class StanleyControllerNode(Node):

    def __init__(self):
        super().__init__('stanley_node')

        # --- ROS 2 Parameters ---
        self.declare_parameter('wheelbase', 0.257)
        self.declare_parameter('k_path', 0.45)
        self.declare_parameter('v_max', 0.3)
        self.declare_parameter('a_lat_max', 0.3)
        
        self.wheelbase = self.get_parameter('wheelbase').value
        self.k_path = self.get_parameter('k_path').value
        self.v_max = self.get_parameter('v_max').value
        self.a_lat_max = self.get_parameter('a_lat_max').value

        # --- Publishers & Subscribers ---
        self.motor_cmd_pub = self.create_publisher(Twist, '/nav_vel', 1)
        self.pose_sub = self.create_subscription(Odometry, '/ekf_pose', self.stanley_controller_callback, 10)

        # --- State Variables ---
        self.prev_i = 0

        # --- Map Configuration ---
        use_small_map = False
        left_hand_traffic = False
        node_sequence = [10, 2, 4, 14, 20, 22, 9, 7, 14, 20, 22, 10]
        
        roadmap = SDCSRoadMap(
            leftHandTraffic=left_hand_traffic,
            useSmallMap=use_small_map
        )

        # Generate, scale, and rotate path
        raw_path = roadmap.generate_path(nodeSequence=node_sequence)[:, :] 
        raw_path[0,:] , raw_path[1,:]=raw_path[0,:] * 0.957 +0.035 ,raw_path[1,:] * 0.954 +0.06
        
        rotated_path = self.rotate_waypoints(raw_path, -6)
        
        # Generate smooth trajectory
        x, y = rotated_path[0, :], rotated_path[1, :]
        x_smooth, y_smooth, yaw, speed, k = self.generate_mpc_trajectory(x, y, v_max=self.v_max, a_lat_max=self.a_lat_max)

        # Stack into [N, 4] array: [x, y, speed, yaw]
        self.waypoints = np.vstack([x_smooth, y_smooth, speed, yaw]).T
        
        self.get_logger().info(f"Stanley Controller Initialized. Tracking {len(self.waypoints)} waypoints.")


    def rotate_waypoints(self, xy, theta_deg):
        """Rotates (N,2) waypoints by theta degrees."""
        theta = np.deg2rad(theta_deg)
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        return R @ xy
    

    def generate_mpc_trajectory(self, x_points, y_points, ds=0.05, v_max=0.3, a_lat_max=0.2):
        """Takes rough waypoints and returns a smooth trajectory (x, y, yaw, v_ref, k)."""
        # 1. Calculate distance along the path (s)
        dx = np.diff(x_points)
        dy = np.diff(y_points)
        distances = np.sqrt(dx**2 + dy**2)
        
        s = np.zeros(len(x_points))
        s[1:] = np.cumsum(distances)
        
        # 2. Create Cubic Spline functions for x(s) and y(s)
        cs_x = CubicSpline(s, x_points)
        cs_y = CubicSpline(s, y_points)
        
        # 3. Interpolate densely
        s_new = np.arange(0, s[-1], ds)
        x_new = cs_x(s_new)
        y_new = cs_y(s_new)
        
        # 4. Calculate Yaw (heading) and Curvature (k)
        dx_ds = cs_x(s_new, 1)
        dy_ds = cs_y(s_new, 1)
        ddx_ds = cs_x(s_new, 2)
        ddy_ds = cs_y(s_new, 2)
        
        yaw = np.arctan2(dy_ds, dx_ds)
        yaw = self.normalize_angle(yaw)
        
        # Curvature k
        k = (dx_ds * ddy_ds - dy_ds * ddx_ds) / np.clip((dx_ds**2 + dy_ds**2)**1.5, 1e-9, None)

        # ---- SPEED PROFILE ----
        v_ref = np.ones_like(x_new) * v_max
        eps = 1e-3
        v_curve = np.sqrt(a_lat_max / (np.abs(k) + eps))
        v_ref = np.minimum(v_curve, v_max)

        # Endpoint Constraint
        v_ref[-1] = 0.0  

        # Backward Pass (Deceleration Ramp)
        a_decel_max = 0.1
        for i in range(len(v_ref) - 2, -1, -1):
            dist = s_new[i+1] - s_new[i]
            allowed_v = np.sqrt(v_ref[i+1]**2 + 2 * a_decel_max * dist)
            v_ref[i] = np.minimum(v_ref[i], allowed_v)

        # Forward Pass (Acceleration Ramp)
        v_ref = self.apply_speed_limits(v_ref, a_max=0.08, ds=ds)
        
        return x_new, y_new, yaw, v_ref, k
    

    def apply_speed_limits(self, v, a_max=0.1, ds=0.1):
        v_smooth = v.copy()
        for i in range(1, len(v)):
            v_smooth[i] = min(v_smooth[i], np.sqrt(v_smooth[i-1]**2 + 2 * a_max * ds))
        return v_smooth


    def stanley_controller_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        v = msg.twist.twist.linear.x
        
        # Orientation (Quaternion -> Euler)
        q = msg.pose.pose.orientation
        roll, pitch, yaw = self.euler_from_quaternion([q.x, q.y, q.z, q.w])

        vehicle_state = np.array([x, y, yaw, v])
        
        # Update dynamic parameter just in case it was changed via rqt
        current_k_path = self.get_parameter('k_path').value 

        steering_angle, speed = self.compute_steering(vehicle_state, self.waypoints, current_k_path)

        cmd = Twist()
        cmd.linear.x = float(speed)
        cmd.angular.z = float(steering_angle)
        self.motor_cmd_pub.publish(cmd)


    def compute_steering(self, vehicle_state, waypoints, k_path):
        theta_e, ef, target_index, goal_velocity = self.calc_theta_and_ef(vehicle_state, waypoints, self.prev_i)
        self.prev_i = target_index

        # Stop condition
        if target_index > len(waypoints) - 3:
            self.get_logger().info('End reached. Stopping vehicle.')
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.motor_cmd_pub.publish(cmd)
            raise SystemExit # Clean exit caught in main()

        # Calculate final steering angle: delta = theta_e + arctan(k * ef / v)
        # We add a small softening constant to velocity to prevent division by zero or jitter at 0 m/s
        k_soft = 0.01 
        cte_front = math.atan2(k_path * ef, abs(vehicle_state[3]) + k_soft)
        delta = self.normalize_angle(cte_front + theta_e)

        return delta, goal_velocity
    

    def calc_theta_and_ef(self, vehicle_state, waypoints, prev_i):
        # Distance to the closest point to the front axle center
        fx = vehicle_state[0] + self.wheelbase * np.cos(vehicle_state[2])
        fy = vehicle_state[1] + self.wheelbase * np.sin(vehicle_state[2])
        position_front_axle = np.array([fx, fy])
        
        nearest_point_front, _, _, target_index = self.nearest_point(
            position_front_axle, waypoints[:, 0:2], prev_i, window=10)
        
        vec_dist_nearest_point = position_front_axle - nearest_point_front

        # Crosstrack error calculation
        front_axle_vec_rot_90 = np.array([
            [np.cos(vehicle_state[2] - np.pi / 2.0)],
            [np.sin(vehicle_state[2] - np.pi / 2.0)]
        ])
        ef = np.dot(vec_dist_nearest_point.T, front_axle_vec_rot_90)[0] # Extract scalar

        # Heading error
        theta_raceline = waypoints[target_index, 3]
        theta_e = self.normalize_angle(theta_raceline - vehicle_state[2])

        # Target velocity
        goal_velocity = waypoints[target_index, 2]

        return theta_e, ef, target_index, goal_velocity
    

    def nearest_point(self, point, trajectory, prev_i, window=10):
        if prev_i is None or prev_i < 0:
            start_idx = 0
            end_idx = trajectory.shape[0] - 1
        else:
            start_idx = max(0, prev_i - window)
            end_idx = min(trajectory.shape[0] - 1, prev_i + window)
            
        traj_window = trajectory[start_idx:end_idx+1, :]
        
        diffs = traj_window[1:, :] - traj_window[:-1, :]
        l2s = diffs[:, 0]**2 + diffs[:, 1]**2
        l2s[l2s == 0] = 1e-9 
        
        dots = np.sum((point - traj_window[:-1, :]) * diffs, axis=1)
        t = np.clip(dots / l2s, 0.0, 1.0)
        
        projections = traj_window[:-1, :] + (t[:, np.newaxis] * diffs)
        dists = np.linalg.norm(point - projections, axis=1)
        
        local_min_idx = np.argmin(dists)
        global_min_dist_segment = start_idx + local_min_idx
        
        return projections[local_min_idx], dists[local_min_idx], t[local_min_idx], global_min_dist_segment
        

    def normalize_angle(self, angle):
        """
        Fast mathematical modulo operation to bound angles strictly to [-pi, pi).
        Replaces slow while-loops.
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi


    def euler_from_quaternion(self, q):
        r = Rot.from_quat(q)
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        return roll, pitch, yaw


def main(args=None):
    rclpy.init(args=args)
    node = StanleyControllerNode()
    try:
        rclpy.spin(node)
    except SystemExit:                 
        node.get_logger().info('Node shut down gracefully via SystemExit.')
    except KeyboardInterrupt:
        node.get_logger().info('Node interrupted via Keyboard.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()