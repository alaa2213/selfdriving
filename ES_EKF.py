import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import numpy as np
import tf2_ros
import math

# A helper function to get yaw from a Quaternion
from tf_transformations import euler_from_quaternion, quaternion_from_euler
class ESEKFNode(Node):
    def __init__(self):
        super().__init__('es_ekf_node')

        # State vector: [px, py, theta, vx, vy, w]
        self.state_ = np.zeros(6)
        # Error state covariance matrix
        self.P_ = np.identity(6) * 0.01

        # Timestamps for delta_t calculation
        self.last_imu_time_ = self.get_clock().now()

        # ROS 2 Subscriptions
        self.imu_sub = self.create_subscription(
            Imu,
            '/carla/ego_vehicle/imu',
            self.imu_callback,
            10
        )
        self.gnss_sub = self.create_subscription(
            PoseStamped,
            '/carla/ego_vehicle/gnss',
            self.gnss_callback,
            10
        )

        # ROS 2 Publishers and TF Broadcaster
        self.odom_pub = self.create_publisher(Odometry, '/odometry/filtered', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

    def imu_callback(self, msg):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_imu_time_).nanoseconds / 1e9
        self.last_imu_time_ = current_time

        if dt > 0.0:
            # Get measurements from the IMU message
            accel_x = msg.linear_acceleration.x
            accel_y = msg.linear_acceleration.y
            gyro_z = msg.angular_velocity.z

            self.predict(dt, accel_x, accel_y, gyro_z)
            self.publish_odometry(msg.header.stamp)

    def gnss_callback(self, msg):
        # Perform the correction step here
        x_gnss = msg.pose.position.x
        y_gnss = msg.pose.position.y
        orientation_q = msg.pose.orientation
        _, _, yaw_gnss = euler_from_quaternion([
            orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w
        ])

        z_gnss = np.array([x_gnss, y_gnss, yaw_gnss])
        R_gnss = np.diag([
            msg.pose.covariance[0], # x
            msg.pose.covariance[7], # y
            msg.pose.covariance[35] # yaw
        ])

        self.update_gnss(z_gnss, R_gnss)
        self.publish_odometry(msg.header.stamp)
        
    def predict(self, dt, accel_x, accel_y, gyro_z):
        # --- 1. Propagate the True State ---
        theta = self.state_[2]
        vx = self.state_[3]
        vy = self.state_[4]
        
        # This is a simplified motion model; a real one might be more complex
        self.state_[0] += (vx * math.cos(theta) - vy * math.sin(theta)) * dt
        self.state_[1] += (vx * math.sin(theta) + vy * math.cos(theta)) * dt
        self.state_[2] += gyro_z * dt
        self.state_[3] += accel_x * dt
        self.state_[4] += accel_y * dt
        self.state_[5] = gyro_z # Assuming gyro_z is already the angular rate

        # --- 2. Propagate the Error Covariance ---
        # F matrix (Jacobian of the error state propagation)
        F = np.identity(6)
        # You need to derive the proper F matrix from your motion model
        # For an ES-EKF, it's the Jacobian of the error state dynamics
        
        # Q matrix (Process noise covariance)
        # Tune these values based on sensor noise characteristics
        Q = np.diag([0.1, 0.1, 0.01, 0.1, 0.1, 0.01])
        Q *= dt # Scale with time step

        self.P_ = F @ self.P_ @ F.T + Q

    def update_gnss(self, z, R):
        # --- 1. Measurement Prediction and Innovation ---
        # The measurement model 'h' is the identity function for this case
        h = self.state_[:3]
        innovation = z - h
        
        # Normalize yaw to be within [-pi, pi]
        innovation[2] = math.atan2(math.sin(innovation[2]), math.cos(innovation[2]))

        # H matrix (Jacobian of the measurement model)
        H = np.zeros((3, 6))
        H[0, 0] = 1.0 # px
        H[1, 1] = 1.0 # py
        H[2, 2] = 1.0 # theta

        # --- 2. Innovation Covariance and Kalman Gain ---
        S = H @ self.P_ @ H.T + R
        K = self.P_ @ H.T @ np.linalg.inv(S)

        # --- 3. Correct the State and Covariance ---
        self.state_ += K @ innovation
        self.P_ = (np.identity(6) - K @ H) @ self.P_

        # Normalize yaw
        self.state_[2] = math.atan2(math.sin(self.state_[2]), math.cos(self.state_[2]))

    def publish_odometry(self, stamp):
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"

        odom_msg.pose.pose.position.x = self.state_[0]
        odom_msg.pose.pose.position.y = self.state_[1]
        odom_msg.pose.pose.position.z = 0.0

        q = quaternion_from_euler(0, 0, self.state_[2])
        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]

        odom_msg.twist.twist.linear.x = self.state_[3]
        odom_msg.twist.twist.linear.y = self.state_[4]
        odom_msg.twist.twist.angular.z = self.state_[5]

        # Populate the covariance matrix
        odom_msg.pose.covariance = self.P_.flatten().tolist()
        odom_msg.twist.covariance = np.zeros(36).flatten().tolist() # Twist covariance for this example

        self.odom_pub.publish(odom_msg)

        # Publish the TF transform
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.state_[0]
        t.transform.translation.y = self.state_[1]
        t.transform.translation.z = 0.0
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    ekf_node = ESEKFNode()
    rclpy.spin(ekf_node)
    ekf_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()