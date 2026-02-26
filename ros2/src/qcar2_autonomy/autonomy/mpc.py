#!/usr/bin/env python3

#  ROS Imports 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point,PoseStamped
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as Rot
# Python Imports
import matplotlib.pyplot as plt
import numpy as np
import pdb
import copy
from threading import local
import cvxpy
import math
import sys
import os
import scipy
from scipy import interpolate
from scipy.interpolate import splprep, splev

from scipy.interpolate import CubicSpline
from hal.products.mats import SDCSRoadMap
from geometry_msgs.msg import Twist



NX                      =       4                               # x = x, y, v, yaw
NU                      =       2                               # a = [accel, steer]
T                       =       10                  # horizon length

# mpc parameters
R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([0.01, 2])  # input difference cost matrix
Q = np.diag([5.0, 5.0, 0.5, 2])  # state cost matrix
Qf = Q  # state final matrix



# Iterative paramter
MAX_ITER                =       1                               # Max iteration
DU_TH                   =       0.1                             # iteration finish param
TARGET_SPEED            =       0.2                           # [m/s] target speed
N_IND_SEARCH            =       10                              # Search index number
DT                      =       0.05                             # [s] time tick

# Vehicle parameters
LENGTH                  =       0.48                            # [m] 
WIDTH                   =       0.268                           # [m] 
BACKTOWHEEL             =       0.07                            # [m] 0.1
WHEEL_LEN               =       0.07                            # [m] 0.1
WHEEL_WIDTH             =       0.07                            # [m]  0.05
TREAD                   =       0.5                             # [m]
WB                      =       0.257                            # [m]
MAX_STEER               =       np.deg2rad(10.0)                # maximum steering angle [rad]
MAX_DSTEER              =       np.deg2rad(3.0)                 # maximum steering speed [rad/s]
MAX_SPEED               =       0.3                            # maximum speed [m/s]
MIN_SPEED               =      0                               # minimum speed [m/s]
MAX_ACCEL               =       0.3                             # maximum accel [m/ss]


class State:
    """
    Vehicle State Class
    """
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.predelta = None


class MPC(Node):
    def __init__(self):
        super().__init__("MPC")
        # Configuration parameters for example script.
        useSmallMap = False
        leftHandTraffic = False
        nodeSequence = [10, 2, 4, 14 , 20 , 22 , 9 , 13 , 19 , 17 , 20 , 22 , 10 ]
        # nodeSequence = [10, 2, 4, 14 , 20 , 22 , 9 , 7 , 14 , 20 , 22 , 10]
        # Create a SDCSRoadMap instance with desired configuration.
        roadmap = SDCSRoadMap(
            leftHandTraffic=leftHandTraffic,
                useSmallMap=useSmallMap
        )
    # Generate the shortest path passing through the given sequence of nodes.
        # - nodeSequence can be a list or tuple of node indicies.
        # - The generated path takes the form of a 2xn numpy array
        path = roadmap.generate_path(nodeSequence=nodeSequence)*0.945
        path=self.rotate_waypoints(path,-7.7)
        x,y=path[0,:],path[1,:]
        x,y,cyaw,speed,k=self.generate_mpc_trajectory(x,y)

        cyaw=self.angle_mod(cyaw)


        self.last_time=self.get_clock().now()

        
        self.waypoints=np.array([x,y,cyaw,speed])
     

        self.initialize=True


        self.cx                              =       self.waypoints[0,:].tolist()
        self.cy                              =       self.waypoints[1,:].tolist()
        self.sp                              =       self.waypoints[3,:].tolist()
        self.cyaw                            =       self.waypoints[2,:] # self.cyaw%(2*math.pi)
        self.cyaw[self.cyaw<0]               =       self.cyaw[self.cyaw<0] + 2*math.pi 
        # self.cyaw                            =       self.cyaw.tolist()
        self.ck                              =       self.waypoints[:,3].tolist()   #kuch bhi  

        

    
        self.global_pub                   =       self.create_publisher(MarkerArray, "global_path_viz", 10)

        self.horizon_pub                =       self.create_publisher(MarkerArray, "horizon_path_pub", 10)

        self.publish=True

        # self.publish_global_arrows()

        self.pose_sub = self.create_subscription(
            Odometry,
            '/ekf_pose',
            self.pose_callback,
            1
        )

        # Publisher: Control Commands
        self.cmd_pub = self.create_publisher(
            Twist,
            '/nav_vel',
            10
        )




    def publish_global_arrows(self):
        # 1. Create the container
        marker_array = MarkerArray()
        
        # 2. Delete old markers (Optional but recommended to prevent "ghosting")
        # Creating a marker with action=DELETEALL handles cleanup
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        # 3. Loop through your trajectory
        for i in range(self.waypoints.shape[1]):
            arrow = Marker()
            arrow.header.frame_id = "/map"
            arrow.header.stamp = self.get_clock().now().to_msg()
            
            # self.get_logger().info(f"shape of wayponits is {self.waypoints.shape}")

            # UNIQUE ID is critical for MarkerArrays!
            arrow.id = i  
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD

            # --- SCALE ---
            # x = Length, y = Shaft Width, z = Head Height
            arrow.scale.x = 0.1  # Make it long enough to see direction
            arrow.scale.y = 0.02
            arrow.scale.z = 0.02
            # --- COLOR ---
            arrow.color.g = 1.0
            arrow.color.a = 1.0

            # --- POSITION ---
            arrow.pose.position.x = self.waypoints[0,i]
            arrow.pose.position.y = self.waypoints[1,i]
            arrow.pose.position.z = 0.0

            # --- ORIENTATION (Yaw -> Quaternion) ---
            # Get the yaw for this specific point
            yaw = self.waypoints[2,i]

            # Simple conversion: Yaw to Quaternion (assuming roll=pitch=0)
            # q_z = sin(yaw/2), q_w = cos(yaw/2)
            arrow.pose.orientation.x = 0.0
            arrow.pose.orientation.y = 0.0
            arrow.pose.orientation.z = math.sin(yaw / 2.0)
            arrow.pose.orientation.w = math.cos(yaw / 2.0)

            marker_array.markers.append(arrow)

        # 4. Publish the whole array at once
        self.global_pub.publish(marker_array)



    def publish_horizon_arrows(self,xref):
        # 1. Create the container

        horizon_array = MarkerArray()
        
        # 2. Delete old markers (Optional but recommended to prevent "ghosting")
        # Creating a marker with action=DELETEALL handles cleanup
        # delete_marker = Marker()
        # delete_marker.action = Marker.DELETEALL
        # marker_array.markers.append(delete_marker)

        # 3. Loop through your trajectory
        # self.get_logger().info(f"shape of xref is {xref.shape}")
        for i in range(xref.shape[1]):
            # self.get_logger().info(f"published the point  {xref[:,i]}")
            arrow = Marker()
            arrow.header.frame_id = "/map"
            arrow.header.stamp = self.get_clock().now().to_msg()
            
            # UNIQUE ID is critical for MarkerArrays!
            arrow.id = i  
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD

            # --- SCALE ---
            # x = Length, y = Shaft Width, z = Head Height
            arrow.scale.x = 0.1  # Make it long enough to see direction
            arrow.scale.y = 0.02
            arrow.scale.z = 0.02
            # --- COLOR ---
            arrow.color.r = 1.0
            arrow.color.a = 1.0

            # --- POSITION ---
            arrow.pose.position.x = xref[0,i]
            arrow.pose.position.y = xref[1,i]
            arrow.pose.position.z = 0.0

            # --- ORIENTATION (Yaw -> Quaternion) ---
            # Get the yaw for this specific point
            yaw = xref[3,i]

            # Simple conversion: Yaw to Quaternion (assuming roll=pitch=0)
            # q_z = sin(yaw/2), q_w = cos(yaw/2)
            arrow.pose.orientation.x = 0.0
            arrow.pose.orientation.y = 0.0
            arrow.pose.orientation.z = math.sin(yaw / 2.0)
            arrow.pose.orientation.w = math.cos(yaw / 2.0)

            horizon_array.markers.append(arrow)

        # 4. Publish the whole array at once
        self.horizon_pub.publish(horizon_array)



        



    def pose_callback(self,msg):

        now=self.get_clock().now()
        dt =(now-self.last_time).nanoseconds * 1e-9
        self.last_time=now

        # self.get_logger().info(f'dt is {now.nanoseconds * 1e-9}')



        dl              =   0.05

        pose_stamp=msg.header.stamp
        pose_time=pose_stamp.sec + pose_stamp.nanosec * 1e-9
        # self.get_logger().info(f"pose time is {pose_time}")




    
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
        # delta = msg.twist.twist.linear.z     #changed in ekf node c++
        beta = 0.0


        state           =   State(x,y,yaw,v)

        if(self.initialize):
            self.initialize         =   False
            self.target_ind         =   np.argmin(np.linalg.norm(self.waypoints[:,:2] - np.array([x,y]).reshape(1,-1),axis = 1))
            self.target_ind, _      =   self.calc_nearest_index(state, self.cx, self.cy, self.cyaw, self.target_ind)
            self.odelta, self.oa    =   None, None


        self.target_ind, _ = self.calc_nearest_index(state, self.cx, self.cy, self.cyaw, self.target_ind)

        # if self.target_ind > len(self.waypoints[:,0]):
        #     self.get_logger().info('end reached. Stopping from inside the node!')
        #     QCarCommands = Twist()
        #     QCarCommands.linear.x=float(0)
        #     QCarCommands.angular.z=float(0)
        #     self.cmd_pub.publish(QCarCommands)

        #     raise SystemExit

        if(self.target_ind > len(self.cx) - T):
            self.target_ind = 0


        xref, self.target_ind, dref = self.calc_ref_trajectory(state, self.cx, self.cy, self.cyaw, self.ck, self.sp, dl, self.target_ind)

        if self.publish==True:
            self.publish=True
        
            self.publish_horizon_arrows(xref)

        x0=[state.x,state.y,state.v,state.yaw]    #current state
        self.oa, self.odelta, ox, oy, oyaw, ov = self.iterative_linear_mpc_control(
            xref, x0, dref, self.oa, self.odelta)
        


        if self.odelta is not None:
            # print("Publishing")
            # print("di: ", di)
            # print("ai: ", ai)
            # print("ov: ", ov[0])
            di, ai = self.odelta[2], self.oa[2]
            self.old_input              =   di
            try:
               

                # Publish Twist Message
                # self.get_logger().info(f" The steering anglre is {np.rad2deg(self.odelta[0])}")
                cmd = Twist()
                cmd.linear.x = state.v+(ai*DT)
                cmd.angular.z = float(di) # Assuming angular.z controls steering
                self.cmd_pub.publish(cmd)
                # now=self.get_clock().now()
                # self.get_logger().info(f"The time difference is {now.nanoseconds* 1e-9 - pose_time}")

            except Exception as e:
                self.get_logger().error(f"MPC Solver Failed: {e}")
                # Safety stop
                self.cmd_pub.publish(Twist())


        
          

        

    
        

    def iterative_linear_mpc_control(self,xref, x0, dref, oa, od):
        """
        MPC contorl with updating operational point iteraitvely
        """
        if oa is None or od is None:
            oa                       =       [0.0] * T
            od                       =       [0.0] * T
        for i in range(MAX_ITER):
            xbar        = self.predict_motion(x0, oa, od, xref)
            poa, pod                 =       oa[:], od[:]
            oa, od, ox, oy, oyaw, ov =       self.linear_mpc_control(xref, xbar, x0, dref)
            du                       =       sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
            if du <= DU_TH:
                break
        # else:
        #     print("Iterative is max iter")
        return oa, od, ox, oy, oyaw, ov
    

    def predict_motion(self,x0, oa, od, xref):
        xbar             =   xref * 0.0
        for i, _ in enumerate(x0):
            xbar[i, 0]   =   x0[i]
        state            =   State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])

        for (ai, di, i) in zip(oa, od, range(1, T + 1)):
            state        = self.update_state(state, ai, di)
            xbar[0, i]   = state.x
            xbar[1, i]   = state.y
            xbar[2, i]   = state.v
            xbar[3, i]   = state.yaw
        return xbar
    

    def update_state(self,state, a, delta):
        # input check
        if delta >= MAX_STEER:
            delta = MAX_STEER

        elif delta <= -MAX_STEER:
            delta = -MAX_STEER

        state.x     =   state.x + state.v * math.cos(state.yaw) * DT
        state.y     =   state.y + state.v * math.sin(state.yaw) * DT
        state.yaw   =   state.yaw + state.v / WB * math.tan(delta) * DT
        state.v     =   state.v + a * DT

        if state.v > MAX_SPEED:
            state.v = MAX_SPEED

        elif state.v < MIN_SPEED:
            state.v = MIN_SPEED

        return state
    


    def linear_mpc_control(self,xref, xbar, x0, dref):
        """
        Linear MPC control
        xref: reference point
        xbar: operational point
        x0: initial state
        dref: reference steer angle
        """
        x               = cvxpy.Variable((NX, T + 1))
        u               = cvxpy.Variable((NU, T))
        cost            = 0.0
        constraints     = []
        for t in range(T):
            cost += cvxpy.quad_form(u[:, t], R)
            if t != 0:
                cost    += cvxpy.quad_form(xref[:, t] - x[:, t], Q)
            A, B, C      = self.get_linear_model_matrix(xbar[2, t], xbar[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]
            
            if t < (T - 1):
                cost        += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
                constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= MAX_DSTEER * DT]
        
        cost            +=  cvxpy.quad_form(xref[:, T] - x[:, T], Qf)
        constraints     +=  [x[:, 0] == x0]
        constraints     +=  [x[2, :] <= MAX_SPEED]
        constraints     +=  [x[2, :] >= MIN_SPEED]
        constraints     +=  [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
        constraints     +=  [cvxpy.abs(u[1, :]) <= MAX_STEER]
        prob             =  cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.ECOS, verbose=False)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox          =   self.get_nparray_from_matrix(x.value[0, :])
            oy          =   self.get_nparray_from_matrix(x.value[1, :])
            ov          =   self.get_nparray_from_matrix(x.value[2, :])
            oyaw        =   self.get_nparray_from_matrix(x.value[3, :])
            oa          =   self.get_nparray_from_matrix(u.value[0, :])
            odelta      =   self.get_nparray_from_matrix(u.value[1, :])
        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None
        return oa, odelta, ox, oy, oyaw, ov
    
    def get_nparray_from_matrix(self,x):
        return np.array(x).flatten()

    def get_linear_model_matrix(self,v, phi, delta):
        A           =   np.zeros((NX, NX))
        A[0, 0]     =   1.0
        A[1, 1]     =   1.0
        A[2, 2]     =   1.0
        A[3, 3]     =   1.0
        A[0, 2]     =   DT * math.cos(phi)
        A[0, 3]     =   - DT * v * math.sin(phi)
        A[1, 2]     =   DT * math.sin(phi)
        A[1, 3]     =   DT * v * math.cos(phi)
        A[3, 2]     =   DT * math.tan(delta) / WB
        B           =   np.zeros((NX, NU))
        B[2, 0]     =   DT
        B[3, 1]     =   DT * v / (WB * math.cos(delta) ** 2)
        C           =   np.zeros(NX)
        C[0]        =   DT * v * math.sin(phi) * phi
        C[1]        =   - DT * v * math.cos(phi) * phi
        C[3]        =   - DT * v * delta / (WB * math.cos(delta) ** 2)

        return A, B, C

    


        







    def calc_ref_trajectory(self,state, cx, cy, cyaw, ck, sp, dl, pind):
        xref        =   np.zeros((NX, T + 1))
        dref        =   np.zeros((1, T + 1))
        ncourse     =   len(cx)
        tref        =   cyaw[pind]
        
        ind, _      =   self.calc_nearest_index(state, cx, cy, cyaw, pind)
        if pind >= ind:
            ind = pind

        xref[0, 0]  =   cx[ind]
        xref[1, 0]  =   cy[ind]
        xref[2, 0]  =   sp[ind]
        xref[3, 0]  =   cyaw[ind]
        dref[0, 0]  =   0.0  # steer operational point should be 0
        travel = 0.0
        
        if(abs(state.yaw - xref[3,0]) > 3.14):
            if(state.yaw < xref[3,0]):
                state.yaw += 2*math.pi
            else:
                print("Hey you out there in the cold")
                xref[3,0] += 2*math.pi
        travel          +=  abs(state.v) * DT
        dind            =   int(round(travel / dl))
        for i in range(1,T + 1):
            
    
            if (ind + dind) < ncourse:
                xref[0, i]  =   cx[ind + dind]
                xref[1, i]  =   cy[ind + dind]
                xref[2, i]  =   sp[ind + dind]
                xref[3, i]  =   cyaw[ind + dind]
                dref[0, i]  =   0.0
            else:
                xref[0, i]  =   cx[ncourse - 1]
                xref[1, i]  =   cy[ncourse - 1]
                xref[2, i]  =   sp[ncourse - 1]
                xref[3, i]  =   cyaw[ncourse - 1]
                dref[0, i]  =   0.0
            if(i>0):
                if xref[3,i] < 1.0 and xref[3,i-1] > 6.0:
                    xref[3,i] += 2*math.pi
            dind+=1
        
        for i in range(T-1,-1,-1):
            if (xref[3,i] - xref[3,i+1]) < -math.pi:
                xref[3,i] += 2*math.pi
        if(state.yaw - xref[3,0]) < -math.pi:
            state.yaw += 2*math.pi

        return xref, ind, dref

    

    def calc_nearest_index(self,state, cx, cy, cyaw, pind):
        dx      =   [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
        dy      =   [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]
        d       =   [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
        mind    =   min(d)
        ind     =   d.index(mind) + pind
        mind    =   math.sqrt(mind)
        dxl     =   cx[ind] - state.x
        dyl     =   cy[ind] - state.y
        angle   =   self.pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))

        if angle < 0:
            mind *= -1
        return ind, mind
    
    def pi_2_pi(self,angle):
        while(angle > math.pi):
            angle = angle - 2.0 * math.pi
        while(angle < -math.pi):
            angle = angle + 2.0 * math.pi
        return angle
    


    def euler_from_quaternion(self,q):
        # q = [x, y, z, w]
        r = Rot.from_quat(q)
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        return roll, pitch, yaw
    

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
    
  
    




















    def generate_mpc_trajectory(self,x_points, y_points, ds=0.06,v_max=0.3,a_lat_max=0.2):
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





def main(args=None):
    rclpy.init(args=args)
    mpc_node = MPC()
    try:
        rclpy.spin(mpc_node)
    except SystemExit:                 # <--- Catch the signal here
        rclpy.logging.get_logger("Quitter").info('Done')
    
    finally:
        mpc_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
if __name__ == '__main__':
    main()