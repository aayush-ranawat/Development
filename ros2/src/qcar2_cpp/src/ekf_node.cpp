#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "qcar2_interfaces/msg/motor_commands.hpp"

#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/time.h"

#include <Eigen/Dense>

using namespace std::chrono_literals;

// Helper to wrap angle to [-pi, pi]
double wrap_to_pi(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

// --- EKF Classes ---

class QcarEKF {
public:
    double L = 0.257;
    Eigen::Vector3d xHat;
    Eigen::Matrix3d P, Q, R, I;

    QcarEKF(Eigen::Vector3d x0, Eigen::Matrix3d P0, Eigen::Matrix3d Q_in, Eigen::Matrix3d R_in) 
        : xHat(x0), P(P0), Q(Q_in), R(R_in) {
        I.setIdentity();
    }

    void prediction(double dt, Eigen::Vector2d u) {
        double v = u(0);
        double delta = u(1);
        double theta = xHat(2);

        // Motion Jacobian
        Eigen::Matrix3d F;
        F << 1.0, 0.0, -dt * v * std::sin(theta),
             0.0, 1.0,  dt * v * std::cos(theta),
             0.0, 0.0,  1.0;

        P = F * P * F.transpose() + Q;

        // Kinematic Model Update
        xHat(0) += dt * v * std::cos(theta);
        xHat(1) += dt * v * std::sin(theta);
        xHat(2) += dt * v * std::tan(delta) / L;
        xHat(2) = wrap_to_pi(xHat(2));
    }

    void correction(Eigen::Vector3d y) {
        Eigen::Matrix3d H = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d S = H * P * H.transpose() + R;
        Eigen::Matrix3d K = P * H.transpose() * S.inverse();

        Eigen::Vector3d z = y - xHat;
        z(2) = wrap_to_pi(z(2));

        xHat += K * z;
        xHat(2) = wrap_to_pi(xHat(2));
        P = (I - K * H) * P;
    }
};

class GyroKF {
public:
    Eigen::Vector2d xHat;
    Eigen::Matrix2d P, Q, A, I;
    double R_val;

    GyroKF(Eigen::Vector2d x0, Eigen::Matrix2d P0, Eigen::Matrix2d Q_in, double R_in)
        : xHat(x0), P(P0), Q(Q_in), R_val(R_in) {
        I.setIdentity();
        A << 0.0, -1.0, 0.0, 0.0;
    }

    void prediction(double dt, double u) {
        Eigen::Matrix2d Ad = I + A * dt;
        Eigen::Vector2d B(dt, 0.0);
        
        xHat = Ad * xHat + B * u;
        P = Ad * P * Ad.transpose() + Q;
    }

    void correction(double y) {
        Eigen::RowVector2d H(1.0, 0.0);
        double S = H * P * H.transpose() + R_val;
        Eigen::Vector2d K = P * H.transpose() / S;

        double z = wrap_to_pi(y - xHat(0));
        xHat += K * z;
        xHat(0) = wrap_to_pi(xHat(0));
        P = (I - K * H) * P;
    }
};

// --- ROS2 NODE ---

class EKFNode : public rclcpp::Node {
public:
    EKFNode() : Node("ekf_node_cpp") {
        // Init EKF
        Eigen::Vector3d x0(-1.287, -0.615, -0.78);
        Eigen::Matrix3d P0 = Eigen::Matrix3d::Identity() * 0.1;
        Eigen::Matrix3d Q_car = Eigen::Vector3d(0.01, 0.01, 0.02).asDiagonal();
        Eigen::Matrix3d R_car = Eigen::Vector3d(0.1, 0.1, 0.01).asDiagonal();
        qcar_ekf_ = std::make_unique<QcarEKF>(x0, P0, Q_car, R_car);

        // Init GyroKF
        gyro_kf_ = std::make_unique<GyroKF>(Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity(), Eigen::Matrix2d::Identity() * 0.001, 0.1);

        // Utils
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/ekf_pose", 10);

        // Subs
        sub_motor_ = this->create_subscription<qcar2_interfaces::msg::MotorCommands>(
            "/qcar2_motor_speed_cmd", 10, std::bind(&EKFNode::motor_cb, this, std::placeholders::_1));
        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/qcar2_imu", 10, std::bind(&EKFNode::imu_cb, this, std::placeholders::_1));

        last_time_ = this->get_clock()->now();
        timer_ = this->create_wall_timer(20ms, std::bind(&EKFNode::update_callback, this));
        
        RCLCPP_INFO(this->get_logger(), "C++ EKF Node Started");
    }

private:
    void motor_cb(const qcar2_interfaces::msg::MotorCommands::SharedPtr msg) {

        for (size_t i = 0; i < msg->motor_names.size(); ++i) {
            if (msg->motor_names[i] == "motor_throttle") u_v_ = msg->values[i];
            if (msg->motor_names[i] == "steering_angle") u_delta_ = msg->values[i];
        }
        // RCLCPP_INFO(this->get_logger(), "Linear Velocity: %.2f", u_v_);
    }

    void imu_cb(const sensor_msgs::msg::Imu::SharedPtr msg) {
        gyro_z_ = msg->angular_velocity.z;
    }

    void update_callback() {
        auto now = this->get_clock()->now();
        double dt = (now - last_time_).seconds();
        if (dt <= 0) return;
        last_time_ = now;

        // Prediction
        qcar_ekf_->prediction(dt, Eigen::Vector2d(u_v_, u_delta_));
        gyro_kf_->prediction(dt, gyro_z_);

        // Correction
        try {
            // Correct method name is lookupTransform
            auto t = tf_buffer_->lookupTransform("map", "base_link", tf2::TimePointZero);
            
            double x_m = t.transform.translation.x;
            double y_m = t.transform.translation.y;

            tf2::Quaternion q(t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w);
            double roll, pitch, yaw_m;
            tf2::Matrix3x3(q).getRPY(roll, pitch, yaw_m);

            gyro_kf_->correction(yaw_m);
            qcar_ekf_->correction(Eigen::Vector3d(x_m, y_m, gyro_kf_->xHat(0)));
        } catch (const tf2::TransformException &ex) {
            // Dead reckoning only if TF is missing
        }

        publish_odom(now);
    }

    void publish_odom(rclcpp::Time now) {
        nav_msgs::msg::Odometry msg;
        msg.header.stamp = now;
        msg.header.frame_id = "map";
        msg.child_frame_id = "base_link_ekf";

        msg.pose.pose.position.x = qcar_ekf_->xHat(0);
        msg.pose.pose.position.y = qcar_ekf_->xHat(1);
        msg.pose.pose.position.z = 0.0;

        tf2::Quaternion q;
        q.setRPY(0, 0, qcar_ekf_->xHat(2));
        q.normalize(); // Crucial for RViz normalization error

        msg.pose.pose.orientation.x = q.x();
        msg.pose.pose.orientation.y = q.y();
        msg.pose.pose.orientation.z = q.z();
        msg.pose.pose.orientation.w = q.w();

        // --- 2. Velocity (NEW ADDITION) ---
        // Map the throttle command (u_v_) directly to linear X velocity
        msg.twist.twist.linear.x = u_v_;
        msg.twist.twist.linear.z = u_delta_; 

        // RCLCPP_INFO(this->get_logger(), "Linear Velocity: %.2f", u_v_);
        
        // Optional: It is good practice to also publish the angular velocity from the gyro
        msg.twist.twist.angular.z = gyro_z_;

        odom_pub_->publish(msg);
    }

    // Member variables
    std::unique_ptr<QcarEKF> qcar_ekf_;
    std::unique_ptr<GyroKF> gyro_kf_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Subscription<qcar2_interfaces::msg::MotorCommands>::SharedPtr sub_motor_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::TimerBase::SharedPtr timer_;

    double u_v_ = 0.0, u_delta_ = 0.0, gyro_z_ = 0.0;
    rclcpp::Time last_time_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<EKFNode>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
