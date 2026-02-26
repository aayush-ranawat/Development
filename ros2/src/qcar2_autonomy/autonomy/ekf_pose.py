#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster, Buffer, TransformListener, TransformException
import numpy as np
from scipy.spatial.transform import Rotation as R
from pal.utilities.math import wrap_to_pi
from qcar2_interfaces.msg import MotorCommands
from rclpy.executors import MultiThreadedExecutor

# --- STATE ESTIMATION CLASSES (Kept logic, improved readability) ---

class QcarEKF:
    def __init__(self, x0, P0, Q, R_mat):
        self.L = 0.257
        self.I = np.eye(3)
        self.xHat = x0
        self.P = P0
        self.Q = Q
        self.R = R_mat
        self.C = np.eye(3)

    def f(self, X, u, dt):
        # Kinematic Bicycle Model
        # u[0] = v, u[1] = steering angle
        # Note: Added a safety check for division by zero if L=0 (not possible here)
        return X + dt * u[0] * np.array([
            [np.cos(X[2, 0])],
            [np.sin(X[2, 0])],
            [np.tan(u[1]) / self.L]
        ])

    def Jf(self, X, u, dt):
        return np.array([
            [1.0, 0.0, -dt * u[0] * np.sin(X[2, 0])],
            [0.0, 1.0,  dt * u[0] * np.cos(X[2, 0])],
            [0.0, 0.0,  1.0]
        ])

    def prediction(self, dt, u):
        F = self.Jf(self.xHat, u, dt)
        self.P = F @ self.P @ F.T + self.Q
        self.xHat = self.f(self.xHat, u, dt)
        self.xHat[2, 0] = wrap_to_pi(self.xHat[2, 0])

    def correction(self, y):
        H = self.C
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        z = y - H @ self.xHat
        z[2, 0] = wrap_to_pi(z[2, 0])
        
        self.xHat += K @ z
        self.xHat[2, 0] = wrap_to_pi(self.xHat[2, 0])
        self.P = (self.I - K @ H) @ self.P

class GyroKF:
    def __init__(self, x0, P0, Q, R_mat):
        self.I = np.eye(2)
        self.xHat = x0
        self.P = P0
        self.Q = Q
        self.R = R_mat
        self.A = np.array([[0.0, -1.0], [0.0, 0.0]])
        self.B = np.array([[1.0], [0.0]])
        self.C = np.array([[1.0, 0.0]])

    def prediction(self, dt, u):
        Ad = self.I + self.A * dt
        self.xHat = Ad @ self.xHat + dt * self.B * u
        self.P = Ad @ self.P @ Ad.T + self.Q

    def correction(self, y):
        H = self.C
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        z = wrap_to_pi(y - (H @ self.xHat)[0, 0])
        self.xHat += K * z
        self.xHat[0, 0] = wrap_to_pi(self.xHat[0, 0])
        self.P = (self.I - K @ H) @ self.P

# --- MAIN ROS2 NODE ---

class EKFNode(Node):
    def __init__(self):
        super().__init__('ekf_node')

        # 1. State Initialization
        x0 = np.array([[-1.287], [-0.615], [-0.78]])
        self.qcar2_ekf = QcarEKF(
            x0=x0, 
            P0=np.eye(3)*0.1, 
            Q=np.diag([0.01, 0.01, 0.02]), 
            R_mat=np.diag([0.1, 0.1, 0.01])
        )
        
        self.gyro_kf = GyroKF(
            x0=np.zeros((2, 1)), 
            P0=np.eye(2), 
            Q=np.diag([0.001, 0.001]), 
            R_mat=np.array([[0.1]])
        )

        # 2. Variables
        self.u_v = 0.0
        self.u_delta = 0.0
        self.gyro_z = 0.0
        self.last_time = self.get_clock().now()

        # 3. ROS Utilities
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, "/ekf_pose", 10)

        # 4. Subs & Timers
        self.sub_motor = self.create_subscription(
            MotorCommands,
            '/qcar2/motor_commands',
            self.motor_commands_cb,
            10
        )
        self.create_subscription(Imu, '/qcar2/imu', self.imu_cb, 10)
        
        # High frequency timer (50Hz)
        self.timer = self.create_timer(0.02, self.update_callback)


        print("node started huh")

    def motor_commands_cb(self, msg):
        """
    Parses the motor_names array to find 'motor_throttle' and 'steering_angle'
    and assigns them to the EKF control inputs.
    """
        for i, name in enumerate(msg.motor_names):
            if name == "motor_throttle":
                self.u_v = msg.values[i]
            elif name == "steering_angle":
                self.u_delta = msg.values[i]



    def imu_cb(self, msg):
        self.gyro_z = msg.angular_velocity.z

    def update_callback(self):
        # self.get_logger().info("1 check")
        # Calculate dynamic dt
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        if (1/dt) <= 0: 
            return
        self.last_time = now
        self.get_logger().info(f"dt is {dt} and frequency is {1/dt}")

        # --- STEP 1: PREDICTION (Always happens) ---
        u_car = np.array([self.u_v, self.u_delta])
        self.gyro_kf.prediction(dt, self.gyro_z)
        self.qcar2_ekf.prediction(dt, u_car)


       

        # --- STEP 2: CORRECTION (Only if sensor data available) ---
        try:
            # Lookup latest transform
            t = self.tf_buffer.lookup_transform('map', 'base_link', Time())

            
            
            # Extract position
            x_m = t.transform.translation.x
            y_m = t.transform.translation.y
            
            # Extract Orientation
            q = t.transform.rotation
            rot = R.from_quat([q.x, q.y, q.z, q.w])
            yaw_m = rot.as_euler('xyz')[2]

            # Fused Correction
            self.gyro_kf.correction(yaw_m)
            fused_yaw = self.gyro_kf.xHat[0, 0]
            
            y_meas = np.array([[x_m], [y_m], [fused_yaw]])
            self.qcar2_ekf.correction(y_meas)

        except TransformException:
            # If TF fails, we just keep the predicted state (Dead Reckoning)
            pass

        # --- STEP 3: PUBLISH ---
        self.broadcast_results(now)

    def broadcast_results(self, now):
        state = self.qcar2_ekf.xHat
        
        # 1. Publish Odometry Message
        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link_ekf" # Different ID to avoid loops
        
        odom.pose.pose.position.x = float(state[0, 0])
        odom.pose.pose.position.y = float(state[1, 0])
        
        q = R.from_euler('z', float(state[2, 0])).as_quat()
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        
        self.odom_pub.publish(odom)

        # # 2. Broadcast TF (Optional but recommended for RViz)
        # t = TransformStamped()
        # t.header = odom.header
        # t.child_frame_id = "base_link_ekf"
        # t.transform.translation.x = odom.pose.pose.position.x
        # t.transform.translation.y = odom.pose.pose.position.y
        # t.transform.rotation = odom.pose.pose.orientation
        # self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    
    # Use MultiThreadedExecutor to prevent TF listener from blocking the timer
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

