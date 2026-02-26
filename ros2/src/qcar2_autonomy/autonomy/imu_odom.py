import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import numpy as np

from geometry_msgs.msg import Quaternion, TransformStamped
from tf2_ros import TransformBroadcaster

class AckermannImuEkf(Node):
    def __init__(self):
        super().__init__('ackermann_imu_ekf')

        # 1. Configuration
        self.declare_parameter('imu_topic', '/qcar2_imu')
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('odom_frame', 'odom')
        
        imu_topic = self.get_parameter('imu_topic').value
        
        # 2. State Initialization (Ackermann / Bicycle Model)
        # State X: [x, y, theta, v]
        # v is strictly "forward speed"
        self.X = np.zeros(4) 
        
        # Error State Covariance P (4x4)
        self.P = np.eye(4) * 0.1 
        
        # Process Noise Q (4x4)
        # [x, y, theta, v]
        # We trust our kinematic model, but assume noise in inputs propagates
        self.Q = np.diag([0.0, 0.0, 0.001, 0.01]) 

        # 3. ROS Setup
        self.last_time = None
        self.sub_imu = self.create_subscription(Imu, imu_topic, self.imu_callback, 10)
        self.pub_odom = self.create_publisher(Odometry, self.get_parameter('odom_topic').value, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info("Ackermann IMU EKF Started. Non-holonomic constraint applied.")

    def imu_callback(self, msg):
        current_time = self.get_clock().now()
        
        if self.last_time is None:
            self.last_time = current_time
            return

        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        # --- 1. EXTRACT MEASUREMENTS ---
        # For Ackermann, we ignore Accel Y (lateral). 
        # We assume the car doesn't slide.
        acc_forward = msg.linear_acceleration.x
        omega = msg.angular_velocity.z

        # --- 2. PREDICTION (BICYCLE MODEL) ---
        x, y, theta, v = self.X
        
        # Pre-calc trig
        c = np.cos(theta)
        s = np.sin(theta)

        # A. Update Orientation
        # theta_new = theta + omega * dt
        new_theta = theta + omega * dt
        # Normalize theta to [-pi, pi]
        new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi

        # B. Update Velocity (Scalar forward speed)
        # v_new = v + acc_forward * dt
        new_v = v + acc_forward * dt

        # C. Update Position
        # Uses the AVERAGE heading for better accuracy over the time step
        # x_new = x + v * cos(theta) * dt + 0.5 * acc * cos(theta) * dt^2
        # Simplified Euler integration:
        new_x = x + v * c * dt + 0.5 * acc_forward * c * dt**2
        new_y = y + v * s * dt + 0.5 * acc_forward * s * dt**2

        # Update State Vector
        self.X = np.array([new_x, new_y, new_theta, new_v])

        # --- 3. JACOBIAN UPDATE (ACKERMANN KINEMATICS) ---
        # State: [x, y, theta, v]
        # We need the partial derivatives of the motion model w.r.t the state variables.
        
        F = np.eye(4)

        # d(x_new)/d(theta) = -v * sin(theta) * dt
        F[0, 2] = -new_v * s * dt 
        # d(x_new)/d(v) = cos(theta) * dt
        F[0, 3] = c * dt

        # d(y_new)/d(theta) = v * cos(theta) * dt
        F[1, 2] = new_v * c * dt
        # d(y_new)/d(v) = sin(theta) * dt
        F[1, 3] = s * dt

        # Propagate Covariance
        self.P = F @ self.P @ F.T + self.Q

        # --- 4. PUBLISH ---

     
        self.publish_odometry(msg.header.stamp,omega)

    def publish_odometry(self, stamp,omega):
        x, y, theta, v = self.X
        
        # NEW
        q = self.get_quaternion_from_euler(0, 0, theta)
        quat_msg = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.get_parameter('odom_frame').value
        odom.child_frame_id = self.get_parameter('base_frame').value
        
        # Pose
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation = quat_msg
        
        # Twist 
        # Crucial for Ackermann: Twist is linear X only!
        odom.twist.twist.linear.x = v
        odom.twist.twist.linear.y = 0.0  # Constraint enforced here
        odom.twist.twist.angular.z = omega # Ideally use the raw IMU omega here if available

        # Covariance Mapping (4x4 -> 6x6)
        pose_cov = np.zeros(36)
        pose_cov[0] = self.P[0,0]   # x
        pose_cov[7] = self.P[1,1]   # y
        pose_cov[35] = self.P[2,2]  # theta
        odom.pose.covariance = pose_cov

        self.pub_odom.publish(odom)

        if self.get_parameter('publish_tf').value:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.get_parameter('odom_frame').value
            t.child_frame_id = self.get_parameter('base_frame').value
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.rotation = quat_msg
            self.tf_broadcaster.sendTransform(t)

    def get_quaternion_from_euler(self, roll, pitch, yaw):
        """
        Convert an Euler angle to a quaternion.
        Input: roll, pitch, yaw
        Output: qx, qy, qz, qw
        """
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return [qx, qy, qz, qw]

def main(args=None):
    rclpy.init(args=args)
    node = AckermannImuEkf()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()