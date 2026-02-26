#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Twist, TransformStamped, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from tf2_ros import TransformBroadcaster
import numpy as np
from scipy.spatial.transform import Rotation as R
from qcar2_interfaces.msg import MotorCommands
from pal.utilities.math import wrap_to_pi


# ==========================================
# HELPER CLASS (From nav_to_pose.py)
# ==========================================
# class QcarEKF:
#     def __init__(self, x0, P0, Q, R_cov):
#         # Nomenclature matches nav_to_pose.py
#         self.L = 0.257  # Wheelbase
#         self.xHat = x0
#         self.P = P0
#         self.Q = Q
#         self.R = R_cov

#     def f(self, X, u, dt):
#         # Kinematic Bicycle Model:
#         # X = [x, y, theta]
#         # u[0] = v (speed), u[1] = delta (steering)
#         return X + dt * u[0] * np.array([
#             [np.cos(X[2,0])],
#             [np.sin(X[2,0])],
#             [np.tan(u[1]) / self.L]
#         ])

#     def prediction(self, dt, u):
#         # Update State Estimate (Prediction Step)
#         self.xHat = self.f(self.xHat, u, dt)
        
#         # Wrap theta to +/- pi
#         self.xHat[2] = (self.xHat[2] + np.pi) % (2 * np.pi) - np.pi
#         return


# region: Helper classes for state estimation
class QcarEKF:

    def __init__(self, x0, P0, Q, R):
        # Nomenclature:
        # - x0: initial estimate
        # - P0: initial covariance matrix estimate
        # - Q: process noise covariance matrix
        # - R: observation noise covariance matrix
        # - xHat: state estimate
        # - P: state covariance matrix
        # - L: wheel base of the QCar
        # - C: output matrix

        self.L = 0.257

        self.I = np.eye(3)
        self.xHat = x0
        self.P = P0
        self.Q = Q
        self.R = R

        self.C = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

    # ==============  SECTION A -  Motion Model ====================
    def f(self, X, u, dt):
        # Kinematic Bicycle Model:
        # - X = [x, y, theta]
        # - u[0] = v (speed in [m/s])
        # - u[1] = delta (steering Angle in [rad])
        # - dt: change in time since last update

        return X + dt * u[0] * np.array([
            [np.cos(X[2,0])],
            [np.sin(X[2,0])],
            [np.tan(u[1]) / self.L]
        ])

    # ==============  SECTION B -  Motion Model Jacobian ====================
    def Jf(self, X, u, dt):
        # Jacobian for the kinematic bicycle model (see self.f)

        return np.array([
                [1, 0, -dt*u[0]*np.sin(X[2,0])],
                [0, 1, dt*u[0]*np.cos(X[2,0])],
                [0, 0, 1]
        ])

    # ==============  SECTION C -  Motion Model Prediction ====================
    def prediction(self, dt, u):

        # Update Covariance Estimate
        F = self.Jf(self.xHat, u, dt)
        self.P = F@self.P@np.transpose(F) + self.Q

        # Update State Estimate
        self.xHat = self.f(self.xHat, u, dt)
        # Wrap th to be in the range of +/- pi
        self.xHat[2] = wrap_to_pi(self.xHat[2])

        return

    # ==============  SECTION D -  Measurement correction ====================
    def correction(self, y):

        # Precompute terms that will be used multiple times
        H = self.C
        P_times_HTransposed = self.P @ np.transpose(H)

        S = H @ P_times_HTransposed + self.R
        K = P_times_HTransposed @ np.linalg.inv(S)

        # Wrap z for th to be in the range of +/- pi
        z = (y - H@self.xHat)
        if len(y) > 1:
            z[2] = wrap_to_pi(z[2])
        else:
            z = wrap_to_pi(z)

        self.xHat += K @ z
        # Wrap th to be in the range of +/- pi
        self.xHat[2] = wrap_to_pi(self.xHat[2])

        self.P = (self.I - K@H) @ self.P

        return

# ==========================================
# ODOMETRY NODE
# ==========================================
class QCarOdometry(Node):

    def __init__(self):
        super().__init__('qcar_odometry')

        # --- Parameters ---
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('odom_frame', 'odom')
        self.base_frame = self.get_parameter('base_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value

        # --- State Variables ---
        # Initial State [x, y, theta]
        x0 = np.zeros((3, 1))
        P0 = np.eye(3)
        # Process Noise and Measurement Noise (Placeholder values for EKF structure)
        Q = np.diagflat([0.0001, 0.0001, 0.001]) 
        R_cov = np.diagflat([0.1, 0.1, 0.01])

        self.ekf = QcarEKF(x0, P0, Q, R_cov)

        self.current_speed = 0.0
        self.current_steering = 0.0
        self.current_yaw_rate = 0.0
        
        self.last_time = self.get_clock().now()

        # --- Publishers ---
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # --- Subscribers ---
        # 1. IMU: Requested source for orientation/angular velocity
        self.create_subscription(Imu, '/qcar2_imu', self.imu_callback, 10)

        # 2. Joint States: Used to calculate linear speed (same math as nav_to_pose)
        # Note: Assuming '/qcar_joint' based on standard naming, adjust if it is '/qcar2_joint'
        # self.create_subscription(JointState, '/qcar2_joint', self.joint_callback, 10)

        # 3. Command Velocity: Used to get the steering angle (delta) 
        # Since JointState usually only gives wheel rotation speed, we listen to commands for steering angle
        # self.create_subscription(Twist, '/cmd_vel_nav', self.cmd_callback, 10)





        # #subscription for input motor commands
        self.create_subscription(MotorCommands,"/qcar2_motor_speed_cmd",self.motor_callback,10)

        # --- Timer ---
        # Run at 50Hz (approx 0.02s)
        self.dt = 0.02
        self.create_timer(self.dt, self.update_odometry)

        self.get_logger().info("QCar Odometry Node Started.")




    def motor_callback(self,msg):
        steering = None
        throttle = None
        for name, value in zip(msg.motor_names, msg.values):
            if name == "steering_angle":
                steering = value
            elif name == "motor_throttle":
                throttle = value
        self.current_steering=steering

        # self.get_logger().info(
        #     f"Steering: {steering} rad | Throttle: {throttle} m/s"
        # )
    def imu_callback(self, msg):
        # We use the gyro z (yaw rate) to improve the heading estimation
        self.current_yaw_rate = msg.angular_velocity.z

    def joint_callback(self, msg):
        # Speed calculation formula extracted directly from nav_to_pose.py
        # Logic: (velocity / gear_ratios) * wheel_circumference
        if len(msg.velocity) > 0:
            self.current_speed = (msg.velocity[0]/(720.0*4.0))*((13.0*19.0)/(70.0*30.0))*(2.0*np.pi)*0.033

    def cmd_callback(self, msg):
        # We assume the robot executes the commanded steering angle
        # This maps angular.z command to steering angle delta
        self.current_steering = msg.angular.z

    def update_odometry(self):
        # self.get_logger().info("QCar Odometry updating")

        current_time = self.get_clock().now()
        # dt calculation in case of jitter (optional, relying on timer is usually fine for simple odom)
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time
        
        # --- 1. Prediction Step ---
        # We can either use the bicycle model's calculated yaw rate: (v * tan(delta) / L)
        # OR use the IMU's measured yaw rate. IMU is usually better for actual rotation.
        # Here we mix them: We use the vehicle model for X/Y, but we inject IMU data for Theta if available.
        
        # Input vector u = [v, delta]
        u = [self.current_speed, self.current_steering]

        # Use the class from nav_to_pose to update state
        self.ekf.prediction(self.dt, u)

        # OPTIONAL: Overwrite the model's theta prediction with IMU integration for better accuracy
        # (Comment this out if you strictly want ONLY the bicycle model equation)
        self.ekf.xHat[2,0] += self.current_yaw_rate * self.dt
        self.ekf.xHat[2,0] = (self.ekf.xHat[2,0] + np.pi) % (2 * np.pi) - np.pi

        # --- 2. Extract State ---
        x = self.ekf.xHat[0, 0]
        y = self.ekf.xHat[1, 0]
        theta = self.ekf.xHat[2, 0]

        # --- 3. Publish TF (Odom -> Base_Link) ---
        q = R.from_euler('z', theta).as_quat() # returns [x, y, z, w]

        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

        # --- 4. Publish Odometry Message ---
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        # Position
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = t.transform.rotation

        # Velocity (Linear in body frame, Angular in body frame)
        odom.twist.twist.linear.x = self.current_speed
        odom.twist.twist.angular.z = self.current_steering # OR self.current_yaw_rate

        self.odom_pub.publish(odom)

def main(args=None):
    rclpy.init(args=args)
    node = QCarOdometry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()