##
#
# Ball pose estimator node.
#
# Subscribes to raw ball detections and publishes estimated pose to /ball/pose.
#
##

# standard imports
import argparse
import numpy as np
import threading
import yaml

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64
from geometry_msgs.msg import PoseStamped

# directory imports
import os
import sys

ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)


############################################################################
# ESTIMATOR NODE
############################################################################
GRAVITY = 9.81
COEFF_OF_RESTITUTION = 0.3


class BallEstimatorNode(Node):
  def __init__(self):
    super().__init__("ball_estimator_node")

    # state
    self.ball_pos = np.zeros(3, dtype=np.float64)
    self.ball_vel = np.zeros(3, dtype=np.float64)
    self.pelvis_pos = np.zeros(3, dtype=np.float64)
    self.pelvis_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    self.target_pos = np.zeros(3, dtype=np.float64)
    self.target_time = -1.0

    # Kalman filter state: [x, y, z, vx, vy, vz]
    self.kf_state = np.zeros(6, dtype=np.float64)
    self.kf_P = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])
    self.kf_Q = np.diag([1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2])
    self.kf_R = np.eye(3, dtype=np.float64) * 0.01
    self.kf_initialized = False
    self.kf_last_time: float | None = None

    config_path = os.path.join(
      os.path.dirname(__file__), "..", "configs", "g1_29dof_tasknpoint.yaml"
    )
    with open(config_path) as f:
      config = yaml.safe_load(f)
    goals = {g["name"]: g for g in config["goals"]}
    self.nominal_target_pos_pelvis = np.array(
      goals["right_hand_target"]["vector"], dtype=np.float64
    )
    self.nominal_target_pos = np.zeros(3, dtype=np.float64)

    self.lock = threading.Lock()

    # subscribers
    self.ball_pos_sub = self.create_subscription(
      PoseStamped, "/ball/pose", self.ball_pose_callback, 10
    )
    self.pelvis_pos_sub = self.create_subscription(
      PoseStamped, "/g1_pelvis/pose", self.pelvis_pose_callback, 10
    )

    # publishers
    self.ball_pose_pub = self.create_publisher(PoseStamped, "/ball/target_pose", 10)
    self.ball_target_time = self.create_publisher(Float64, "/ball/target_time", 10)

    self.create_timer(0.02, self.timer_callback)

    print("Ball estimator node initialized.")

  # callback: ball position [x, y, z] in world frame
  def ball_pose_callback(self, msg: PoseStamped):
    z = np.array(
      [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float64
    )
    self.filter_pose(z)

  # callback: pelvis pose in world frame
  def pelvis_pose_callback(self, msg: PoseStamped):
    self.pelvis_pos = np.array(
      [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float64
    )
    self.pelvis_quat = np.array(
      [
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w,
      ],
      dtype=np.float64,
    )
    q_vec = self.pelvis_quat[:3]
    qw = self.pelvis_quat[3]
    t = 2.0 * np.cross(q_vec, self.nominal_target_pos_pelvis)
    self.nominal_target_pos = (
      self.nominal_target_pos_pelvis + qw * t + np.cross(q_vec, t) + self.pelvis_pos
    )

  # kalman filter update: estimate velocity and position of the ball from the measurements
  def filter_pose(self, z: np.ndarray):
    now = self.get_clock().now().nanoseconds * 1e-9

    if not self.kf_initialized:
      self.kf_state[:3] = z
      self.kf_initialized = True
      self.kf_last_time = now
      self.ball_pos = self.kf_state[:3].copy()
      self.ball_vel = self.kf_state[3:].copy()
      return

    dt = now - self.kf_last_time
    self.kf_last_time = now
    if dt <= 0:
      return

    # State transition: position integrates velocity, velocity is constant (gravity handled as input)
    F = np.eye(6, dtype=np.float64)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt

    # Deterministic gravity input
    u = np.array([0.0, 0.0, -0.5 * GRAVITY * dt**2, 0.0, 0.0, -GRAVITY * dt])

    # Predict
    x_pred = F @ self.kf_state + u
    P_pred = F @ self.kf_P @ F.T + self.kf_Q

    # Update: observation is position only
    H = np.zeros((3, 6), dtype=np.float64)
    H[0, 0] = H[1, 1] = H[2, 2] = 1.0

    S = H @ P_pred @ H.T + self.kf_R
    K = P_pred @ H.T @ np.linalg.inv(S)

    self.kf_state = x_pred + K @ (z - H @ x_pred)
    self.kf_P = (np.eye(6) - K @ H) @ P_pred

    self.ball_pos = self.kf_state[:3].copy()
    self.ball_vel = self.kf_state[3:].copy()

  # given the estimated position and velocity, find the point closest to the robot target point in world frame along the ball's trajectory, this will be the target pos
  def estimate_target_point(self):
    """
    Minimize
            || [x + vx*t, y + vy*t, z + vz*t - 0.5*g*t^2] - [x*, y*, z*] ||^2
    over t >= 0.

    Returns
    -------
    t_star : float
            Optimal nonnegative time.
    f_star : float
            Minimum squared distance.
    """
    dq = self.ball_pos - self.nominal_target_pos
    dx, dy, dz = dq
    vx, vy, vz = self.ball_vel

    def f(t):
      xt = dx + vx * t
      yt = dy + vy * t
      zt = dz + vz * t - 0.5 * GRAVITY * t**2
      return xt**2 + yt**2 + zt**2

    # Coefficients of f'(t) = 0:
    # g^2 t^3 - 3 g vz t^2 + 2(vx^2 + vy^2 + vz^2 - g dz)t + 2(dx vx + dy vy + dz vz) = 0
    coeffs = np.array(
      [
        GRAVITY**2,
        -3.0 * GRAVITY * vz,
        2.0 * (vx**2 + vy**2 + vz**2 - GRAVITY * dz),
        2.0 * (dx * vx + dy * vy + dz * vz),
      ],
      dtype=float,
    )

    roots = np.roots(coeffs)
    # Keep only real roots with t >= 0
    real_nonneg = roots[np.isclose(roots.imag, 0.0, atol=1e-10)].real
    candidates = np.concatenate(([0.0], real_nonneg[real_nonneg >= 0.0]))
    values = np.array([f(t) for t in candidates])
    idx = np.argmin(values)

    best_time, best_value = float(candidates[idx]), float(values[idx])
    if (
      best_value > 1
    ):  # if the best value is still large, just go for the nominal target
      self.target_pos = self.nominal_target_pos
      self.target_time = -1.0
    else:
      self.target_pos = (
        self.ball_pos
        + self.ball_vel * best_time
        + np.array([0.0, 0.0, -0.5 * GRAVITY * best_time**2])
      )
      self.target_time = best_time

  def timer_callback(self):
    if not self.kf_initialized:
      return
    self.estimate_target_point()
    self.publish_pose()
    self.publish_target_time()

  def publish_pose(self):
    msg = PoseStamped()
    msg.header.stamp = self.get_clock().now().to_msg()
    msg.header.frame_id = "world"
    msg.pose.position.x = self.target_pos[0]  # in world frame (for all of these)
    msg.pose.position.y = self.target_pos[1]
    msg.pose.position.z = self.target_pos[2]
    msg.pose.orientation.w = 1.0
    msg.pose.orientation.x = 0.0
    msg.pose.orientation.y = 0.0
    msg.pose.orientation.z = 0.0
    self.ball_pose_pub.publish(msg)

  def publish_target_time(self):
    # publish the time until the ball reaches the target point
    msg = Float64()
    msg.data = self.target_time
    self.ball_target_time.publish(msg)


############################################################################
# MAIN FUNCTION
############################################################################


def main(args=None):
  rclpy.init()

  parser = argparse.ArgumentParser(description="Ball pose estimator node.")
  args = parser.parse_args()

  node = BallEstimatorNode()

  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()

  print("Ball estimator shutdown complete.")


if __name__ == "__main__":
  main()
