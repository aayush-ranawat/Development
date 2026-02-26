#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Header

import numpy as np
from hal.products.mats import SDCSRoadMap
from tf2_ros import Buffer, TransformListener, TransformException
from scipy.spatial.transform import Rotation as R # Useful for Quaternion -> Euler
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from qcar2_interfaces.msg import MotorCommands
from scipy.spatial.transform import Rotation as Rot
from scipy.interpolate import CubicSpline
import math

class lqr(Node):

    def __init__(self):

        super().__init__('lqr_node')


        self.wheelbase=0.257
        self.motor_cmd_pub=self.create_publisher(Twist,'/nav_vel', 1)
        self.vehicle_control_e_cog = 0       # e_cg: lateral error of CoG to ref trajectory
        self.vehicle_control_theta_e = 0 



        #Define LQR Matrix and Parameter
        matrix_q_1=0.999
        matrix_q_2=0.0
        matrix_q_3=0.0066
        matrix_q_4=0.0
        matrix_r=0.75
        self.matrix_q = [matrix_q_1, matrix_q_2, matrix_q_3, matrix_q_4]
        self.matrix_r = [matrix_r]



         # ---- CONFIG ----
        useSmallMap = False
        leftHandTraffic = False
        nodeSequence = [10, 2, 4, 14 , 20 , 22 , 10 ]

        roadmap = SDCSRoadMap(
            leftHandTraffic=leftHandTraffic,
            useSmallMap=useSmallMap
        )
        

        # Path shape: 2 x N  (x, y)
        self.path_np = roadmap.generate_path(nodeSequence=nodeSequence)[:2, :] * 0.96      #0.975    #scaling the path

        self.path_np = self.rotate_waypoints(self.path_np,-7.2)                            # rotate by 9 degree
        
        x,y=self.path_np[0,:],self.path_np[1,:]
        x,y,cyaw,speed,k=self.generate_mpc_trajectory(x,y)

        self.waypoints=np.vstack([x,y,speed,cyaw,k]).T
        
        self.pose_sub = self.create_subscription(
            Odometry,
            '/ekf_pose',
            self.lqr_controller,
            10
        )
        

    def lqr_controller(self,msg):
         # 1. Position
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            
            v = msg.twist.twist.linear.x
            # 2. Orientation (Quaternion -> Euler)
            q = msg.pose.pose.orientation
            quat_list = [q.x, q.y, q.z, q.w]
            (roll, pitch, yaw) = self.euler_from_quaternion(quat_list)

            vehicle_state = np.array([x, y, yaw, v])


            timestep ,iterations ,eps= 0.01 ,50 ,0.001
            steering_angle, speed = self.controller(vehicle_state, self.waypoints, timestep, self.matrix_q, self.matrix_r, iterations, eps)

            QCarCommands = Twist()
            QCarCommands.linear.x=float(speed)
            QCarCommands.angular.z=float(steering_angle)
            self.motor_cmd_pub.publish(QCarCommands)




    def controller(self, vehicle_state, waypoints, ts, matrix_q, matrix_r, max_iteration, eps):
        """
        Compute lateral control command.

        Args:
            vehicle_state (numpy.ndarray [4, ]): [x, y, heading, velocity] of the vehicle
            waypoints (numpy.ndarray [N, 5]): waypoints to track
            ts (float): discretization time step
            matrix_q ([float], len=4): weights on the states
            matrix_r ([float], len=1): weights on control input
            max_iteration (int): maximum iteration for solving
            eps (float): error tolerance for solving

        Returns:
            steer_angle (float): desired steering angle
            v_ref (float): desired velocity
        """

        # size of controlled states
        state_size = 4

        # Saving lateral error and heading error from previous timestep
        e_cog_old = self.vehicle_control_e_cog
        theta_e_old = self.vehicle_control_theta_e

        # Calculating current errors and reference points from reference trajectory
        theta_e, e_cg, yaw_ref, k_ref, v_ref = self.calc_control_points(vehicle_state, waypoints)

        #Update the calculation matrix based on the current vehicle state
        matrix_ad_, matrix_bd_ = self.update_matrix(vehicle_state, state_size, ts, self.wheelbase)

        matrix_state_ = np.zeros((state_size, 1))
        matrix_r_ = np.diag(matrix_r)
        matrix_q_ = np.diag(matrix_q)

        matrix_k_ = self.solve_lqr(matrix_ad_, matrix_bd_, matrix_q_, matrix_r_, eps, max_iteration)

        matrix_state_[0][0] = e_cg
        matrix_state_[1][0] = (e_cg - e_cog_old) / ts
        matrix_state_[2][0] = theta_e
        matrix_state_[3][0] = (theta_e - theta_e_old) / ts

        steer_angle_feedback = (matrix_k_ @ matrix_state_)[0][0]

        #Compute feed forward control term to decrease the steady error.
        steer_angle_feedforward = k_ref * self.wheelbase

        # Calculate final steering angle in [rad]
        steer_angle = steer_angle_feedback + steer_angle_feedforward

        return steer_angle, v_ref
    


    def solve_lqr(self,A, B, Q, R, tolerance, max_num_iteration):
        """
        Iteratively calculating feedback matrix K

        Args:
            A: matrix_a
            B: matrix_b
            Q: matrix_q
            R: matrix_r_
            tolerance: lqr_eps
            max_num_iteration: max_iteration

        Returns:
            K: feedback matrix
        """

        M = np.zeros((Q.shape[0], R.shape[1]))

        AT = A.T
        BT = B.T
        MT = M.T

        P = Q
        num_iteration = 0
        diff = math.inf

        while num_iteration < max_num_iteration and diff > tolerance:
            num_iteration += 1
            P_next = AT @ P @ A - (AT @ P @ B + M) @ \
                    np.linalg.pinv(R + BT @ P @ B) @ (BT @ P @ A + MT) + Q

            # check the difference between P and P_next
            diff = np.abs(np.max(P_next - P))
            P = P_next

        K = np.linalg.pinv(BT @ P @ B + R) @ (BT @ P @ A + MT)

        return K
    


    def update_matrix(self,vehicle_state, state_size, timestep, wheelbase):
        """
        calc A and b matrices of linearized, discrete system.

        Args:
            vehicle_state:
            state_size:
            timestep:
            wheelbase:

        Returns:
            A:
            b:
        """

        #Current vehicle velocity
        v = vehicle_state[3]

        #Initialization of the time discrete A matrix
        matrix_ad_ = np.zeros((state_size, state_size))

        matrix_ad_[0][0] = 1.0
        matrix_ad_[0][1] = timestep
        matrix_ad_[1][2] = v
        matrix_ad_[2][2] = 1.0
        matrix_ad_[2][3] = timestep

        # b = [0.0, 0.0, 0.0, v / L].T
        matrix_bd_ = np.zeros((state_size, 1))  # time discrete b matrix
        matrix_bd_[3][0] = v / wheelbase

        return matrix_ad_, matrix_bd_


    

    def calc_control_points(self, vehicle_state, waypoints):
        """
        Calculate the heading and cross-track errors and target velocity and curvature
        Args:
            vehicle_state (numpy.ndarray [4, ]): [x, y, heading, velocity] of the vehicle
            waypoints (numpy.ndarray [N, 5]): waypoints to track [x, y, velocity, heading, curvature]

        Returns:
            theta_e (float): heading error
            e_cog (float): lateral crosstrack error
            theta_raceline (float): target heading
            kappa_ref (float): target curvature
            goal_veloctiy (float): target velocity
        """

        # distance to the closest point to the front axle center
        fx = vehicle_state[0] + self.wheelbase * math.cos(vehicle_state[2])
        fy = vehicle_state[1] + self.wheelbase * math.sin(vehicle_state[2])
        position_front_axle = np.array([fx, fy])
        nearest_point_front, nearest_dist, t, target_index = self.nearest_point(position_front_axle, self.waypoints[:, 0:2])
        vec_dist_nearest_point = position_front_axle - nearest_point_front

        # crosstrack error
        front_axle_vec_rot_90 = np.array([[math.cos(vehicle_state[2] - math.pi / 2.0)],
                                          [math.sin(vehicle_state[2] - math.pi / 2.0)]])
        ef = np.dot(vec_dist_nearest_point.T, front_axle_vec_rot_90)

        # heading error
        # NOTE: If your raceline is based on a different coordinate system you need to -+ pi/2 = 90 degrees
        theta_raceline = waypoints[target_index, 3]
        theta_e = self.pi_2_pi(theta_raceline - vehicle_state[2])

        # target velocity
        goal_veloctiy = waypoints[target_index, 2]

        # reference curvature
        kappa_ref = self.waypoints[target_index, 4]

        # saving control errors
        self.vehicle_control_e_cog = ef[0]
        self.vehicle_control_theta_e = theta_e

        return theta_e, ef[0], theta_raceline, kappa_ref, goal_veloctiy
    


    def nearest_point(self,point, trajectory):
        """
        Return the nearest point along the given piecewise linear trajectory.

        Args:
            point (numpy.ndarray, (2, )): (x, y) of current pose
            trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
                NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world

        Returns:
            nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
            nearest_dist (float): distance to the nearest point
            t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
            i (int): index of nearest point in the array of trajectory waypoints
        """
        diffs = trajectory[1:,:] - trajectory[:-1,:]
        l2s   = diffs[:,0]**2 + diffs[:,1]**2
        dots = np.empty((trajectory.shape[0]-1, ))
        for i in range(dots.shape[0]):
            dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
        t = dots / l2s
        t[t<0.0] = 0.0
        t[t>1.0] = 1.0
        projections = trajectory[:-1,:] + (t*diffs.T).T
        dists = np.empty((projections.shape[0],))
        for i in range(dists.shape[0]):
            temp = point - projections[i]
            dists[i] = np.sqrt(np.sum(temp*temp))
        min_dist_segment = np.argmin(dists)
        return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment
    

    def pi_2_pi(self,angle):
        while(angle > np.pi):
            angle = angle - 2.0 * np.pi
        while(angle < -np.pi):
            angle = angle + 2.0 * np.pi
        return angle
    

            


    def euler_from_quaternion(self,q):
            # q = [x, y, z, w]
            r = Rot.from_quat(q)
            roll, pitch, yaw = r.as_euler('xyz', degrees=False)
            return roll, pitch, yaw
    

    def angle_mod(self,x, zero_2_2pi=False, degree=False):
        """
        Angle modulo operation
        Default angle modulo range is [-pi, pi)

        Parameters
        ----------
        x : float or array_like
            A angle or an array of angles. This array is flattened for
            the calculation. When an angle is provided, a float angle is returned.
        zero_2_2pi : bool, optional
            Change angle modulo range to [0, 2pi)
            Default is False.
        degree : bool, optional
            If True, then the given angles are assumed to be in degrees.
            Default is False.

        Returns
        -------
        ret : float or ndarray
            an angle or an array of modulated angle.

        Examples
        --------
        >>> angle_mod(-4.0)
        2.28318531

        >>> angle_mod([-4.0])
        np.array(2.28318531)

        >>> angle_mod([-150.0, 190.0, 350], degree=True)
        array([-150., -170.,  -10.])

        >>> angle_mod(-60.0, zero_2_2pi=True, degree=True)
        array([300.])

        """
        if isinstance(x, float):
            is_float = True
        else:
            is_float = False

        x = np.asarray(x).flatten()
        if degree:
            x = np.deg2rad(x)

        if zero_2_2pi:
            mod_angle = x % (2 * np.pi)
        else:
            mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

        if degree:
            mod_angle = np.rad2deg(mod_angle)

        if is_float:
            return mod_angle.item()
        else:
            return mod_angle
        


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
    

    def generate_mpc_trajectory(self,x_points, y_points, ds=0.1,v_max=0.5,a_lat_max=0.2):
        """
        Takes rough waypoints and returns a smooth trajectory (x, y, yaw, k)
        ds: distance between interpolated points [m]
        """
        
        # 1. Calculate distance along the path (s)
        dx = np.diff(x_points)
        dy = np.diff(y_points)
        distances = np.sqrt(dx**2 + dy**2)
        
        # Cumulative distance (s)
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
        # First derivatives (dx/ds, dy/ds)
        dx_ds = cs_x(s_new, 1)
        dy_ds = cs_y(s_new, 1)
        
        # Second derivatives (d2x/ds2, d2y/ds2)
        ddx_ds = cs_x(s_new, 2)
        ddy_ds = cs_y(s_new, 2)
        
        # Yaw = arctan2(dy/ds, dx/ds)
        yaw = np.arctan2(dy_ds, dx_ds)
        yaw=self.angle_mod(yaw)
        
        # Curvature k = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
        # Since parameter is arc length s, x'^2 + y'^2 approx 1, simplifying calculation
        k = (dx_ds * ddy_ds - dy_ds * ddx_ds) / ((dx_ds**2 + dy_ds**2)**(1.5))


            # ---- SPEED PROFILE ----
        target_speed=v_max
        v_ref = np.ones_like(x_new) * target_speed
        # eps = 1e-3
        # v_curve = np.sqrt(a_lat_max / (np.abs(k) + eps))
        # v_ref = np.minimum(v_curve, v_max)

        v_ref = self.apply_speed_limits(v_ref, a_max=0.08, ds=ds)

        
        return x_new, y_new, yaw,v_ref,k
    

    def apply_speed_limits(self, v, a_max=0.1, ds=0.1):
        v_smooth = v.copy()
        for i in range(1, len(v)):
            v_smooth[i] = min(v_smooth[i], np.sqrt(v_smooth[i-1]**2 + 2*a_max*ds))
        return v_smooth
    













def main(args=None):
    rclpy.init(args=args)
    node=lqr()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()