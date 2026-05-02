##
#
# Ball estimator visualization node.
#
# Live 3D plot: pelvis, ball, target point, ball history, and estimated trajectory.
#
##

import threading
from collections import deque

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)

GRAVITY = 9.81
HISTORY_LEN = 100
TRAJ_DURATION = 2.0
TRAJ_STEPS = 50


class BallEstimatorPlottingNode(Node):
  def __init__(self):
    super().__init__("ball_estimator_plotting_node")

    self.lock = threading.Lock()
    self.ball_pos: np.ndarray | None = None
    self.pelvis_pos: np.ndarray | None = None
    self.target_pos: np.ndarray | None = None
    self.ball_history: deque[tuple[float, np.ndarray]] = deque(maxlen=HISTORY_LEN)

    self.create_subscription(PoseStamped, "/ball/pose", self.ball_callback, 10)
    self.create_subscription(PoseStamped, "/g1_pelvis/pose", self.pelvis_callback, 10)
    self.create_subscription(PoseStamped, "/ball/target_pose", self.target_callback, 10)

  def ball_callback(self, msg: PoseStamped):
    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    with self.lock:
      self.ball_pos = pos
      self.ball_history.append((t, pos.copy()))

  def pelvis_callback(self, msg: PoseStamped):
    with self.lock:
      self.pelvis_pos = np.array(
        [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
      )

  def target_callback(self, msg: PoseStamped):
    with self.lock:
      self.target_pos = np.array(
        [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
      )

  def get_state(self):
    with self.lock:
      ball = self.ball_pos.copy() if self.ball_pos is not None else None
      pelvis = self.pelvis_pos.copy() if self.pelvis_pos is not None else None
      target = self.target_pos.copy() if self.target_pos is not None else None
      history = list(self.ball_history)
    vel = _estimate_velocity(history)
    return ball, pelvis, target, history, vel


def _estimate_velocity(
  history: list[tuple[float, np.ndarray]],
) -> np.ndarray | None:
  if len(history) < 3:
    return None
  recent = history[-5:] if len(history) >= 5 else history
  times = np.array([h[0] for h in recent])
  positions = np.array([h[1] for h in recent])
  ts = times - times[0]
  # Subtract gravity from z before linear fitting so the fit is unbiased
  pos_corrected = positions.copy()
  pos_corrected[:, 2] += 0.5 * GRAVITY * ts**2
  A = np.column_stack([np.ones_like(ts), ts])
  result = np.linalg.lstsq(A, pos_corrected, rcond=None)
  return result[0][1]  # velocity = slope of the linear fit


def _compute_trajectory(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
  ts = np.linspace(0, TRAJ_DURATION, TRAJ_STEPS)
  traj = np.column_stack(
    [
      pos[0] + vel[0] * ts,
      pos[1] + vel[1] * ts,
      pos[2] + vel[2] * ts - 0.5 * GRAVITY * ts**2,
    ]
  )
  return traj[traj[:, 2] >= 0]


def main():
  rclpy.init()
  node = BallEstimatorPlottingNode()

  ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
  ros_thread.start()

  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111, projection="3d")

  def update(_frame):
    ball, pelvis, target, history, vel = node.get_state()
    ax.cla()
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Ball Estimator")

    if history:
      hist = np.array([h[1] for h in history])
      ax.plot(
        hist[:, 0],
        hist[:, 1],
        hist[:, 2],
        "b.",
        markersize=3,
        alpha=0.5,
        label="Ball history",
      )

    if ball is not None and vel is not None:
      traj = _compute_trajectory(ball, vel)
      if len(traj) > 1:
        ax.plot(
          traj[:, 0],
          traj[:, 1],
          traj[:, 2],
          "c--",
          linewidth=1.5,
          label="Estimated trajectory",
        )

    if ball is not None:
      ax.scatter(*ball, c="green", s=80, marker="o", label="Ball", zorder=5)

    if pelvis is not None:
      ax.scatter(*pelvis, c="orange", s=120, marker="s", label="Pelvis", zorder=5)

    if target is not None:
      ax.scatter(*target, c="red", s=150, marker="*", label="Target", zorder=5)

    center = ball if ball is not None else pelvis
    if center is not None:
      r = 2.0
      ax.set_xlim(center[0] - r, center[0] + r)
      ax.set_ylim(center[1] - r, center[1] + r)
      ax.set_zlim(0, max(center[2] + r, 2.0))

    ax.legend(loc="upper left", fontsize=8)
    return []

  ani = animation.FuncAnimation(fig, update, interval=50, blit=False)  # noqa: F841
  plt.show()

  node.destroy_node()
  rclpy.shutdown()


if __name__ == "__main__":
  main()
