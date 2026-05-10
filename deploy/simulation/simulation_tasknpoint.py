##
#
# Simulation node using Mujoco to simulate the robot.
#
##

# standard imports
import argparse
import time

# mujoco imports
import mujoco
import mujoco.viewer

# other imports
import numpy as np
import yaml

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float32MultiArray

# directory imports
import sys
import os

ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)

# custom imports
from utils.math_utils import (
  quat_conjugate,
  quat_multiply,
  quat_to_rotation_matrix,
  quat_to_rpy,
  rpy_to_quat,
)


############################################################################
# SIMULATION NODE
############################################################################


class SimulationNode(Node):
  """
  Asynchronous simulation node that runs the Mujoco simulation.
  """

  def __init__(self, config_path: str, apply_noise: bool = False):
    super().__init__("simulation_node")

    # load config file
    self.config = self.load_config(config_path)

    # whether to apply per-sensor Gaussian noise
    self.apply_noise = apply_noise

    # load params
    self.init_params()

    # initialize mujoco
    self.init_simulation()

    # compute world-frame goal targets anchored to the robot's initial pose

    self.motion_idx = 0
    self.init_goals()

    # ROS publishers
    self.pelvis_imu_state_pub = self.create_publisher(
      Float32MultiArray, "deploy_robot/pelvis_imu_state", 10
    )
    self.joint_state_pub = self.create_publisher(
      Float32MultiArray, "deploy_robot/joint_state", 10
    )
    self.simulation_time_pub = self.create_publisher(
      Float64, "deploy_robot/simulation_time", 10
    )
    self.goal_pub = self.create_publisher(Float32MultiArray, "deploy_robot/goals", 10)
    self.which_motion_pub = self.create_publisher(
      Float64, "deploy_robot/which_motion", 10
    )

    # ROS subscribers
    self.command_sub = self.create_subscription(
      Float32MultiArray, "deploy_robot/command", self.command_callback, 10
    )
    self.motion_frame_sub = self.create_subscription(
      Float64, "deploy_robot/motion_frame", self.motion_frame_callback, 10
    )
    self.which_motion_sub = self.create_subscription(
      Float32MultiArray, "deploy_robot/joystick", self.which_motion_callback, 10
    )

    # initial command state
    self.command_received = False
    self.qpos_des = np.zeros(self.nu)
    self.qvel_des = np.zeros(self.nu)
    self.tau_ff = np.zeros(self.nu)
    self.Kp = np.zeros(self.nu)
    self.Kd = np.zeros(self.nu)
    self.motion_frame = 0

    # create a timer to run the simulation loop
    sim_period = 0.0  # run as fast as possible, real-time sync is handled in the loop
    self.timer = self.create_timer(sim_period, self.step_simulation)

    # create timers for publishing
    imu_state_period = self.sim_dt
    joint_state_period = self.sim_dt
    goals_state_period = self.sim_dt
    self.pelvis_imu_timer = self.create_timer(imu_state_period, self.publish_pelvis_imu)
    self.joint_timer = self.create_timer(joint_state_period, self.publish_joint_state)
    self.goals_timer = self.create_timer(goals_state_period, self.publish_goals)

    print("Simulation node initialized.")
    print("    Press [Tab] to toggle the left UI.")
    print("    Press [Shift + Tab] to toggle the right UI.")

  #################################################################
  # INITIALIZATION
  #################################################################

  # load the config file
  def load_config(self, config_path: str):
    # open the config file and load it
    config_path_full = ROOT_DIR + "/deploy/configs/" + config_path
    with open(config_path_full, "r") as f:
      config = yaml.safe_load(f)

    return config

  # load policy params
  def init_params(self):
    # set the default state
    self.default_base = np.array(self.config["default_base_pos"])
    self.default_joints = np.array(self.config["default_joint_pos"])

  # initialize the mujoco simulation
  def init_simulation(self):
    # load the XML path
    models_path = ROOT_DIR + "/models/"
    xml_path = models_path + self.config["xml_path"]

    # load the mujoco model
    self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
    self.mj_data = mujoco.MjData(self.mj_model)

    # load model properties
    self.nq = self.mj_model.nq
    self.nv = self.mj_model.nv
    self.nu = self.mj_model.nu
    self.sim_dt = self.mj_model.opt.timestep

    # make sure the default joints are the correct size
    assert len(self.default_joints) == self.nu, (
      f"Default joint angles must be of size {self.nu}, got {len(self.default_joints)}."
    )

    # assign initial state
    self.mj_data.qpos[:7] = self.default_base
    self.mj_data.qpos[7 : 7 + self.nu] = self.default_joints

    # build list of joint sensor names (matching actuator order)
    self.joint_pos_sensor_names = []
    self.joint_vel_sensor_names = []
    for i in range(self.nu):
      joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
      self.joint_pos_sensor_names.append(f"{joint_name}_pos_sensor")
      self.joint_vel_sensor_names.append(f"{joint_name}_vel_sensor")

    print(f"Loaded Mujoco model from [{xml_path}].")
    print(f"    Sim dt: {self.sim_dt} seconds.")
    print(f"    nq: {self.nq}")
    print(f"    nv: {self.nv}")
    print(f"    nu: {self.nu}")

    # launch the viewer
    self.viewer = mujoco.viewer.launch_passive(
      self.mj_model,
      self.mj_data,
      show_left_ui=False,  # disable left tab (use 'Tab' for toggling on/off)
      show_right_ui=False,  # disable right tab (use 'Tab + Shift' for toggling on/off)
    )

    # viewer settings
    self._viewer_font_scale = getattr(
      mujoco.mjtFontScale,
      "mjFONTSCALE_250",
      getattr(
        mujoco.mjtFontScale, "mjFONTSCALE_200", mujoco.mjtFontScale.mjFONTSCALE_150
      ),
    )

    # camera settings
    self.viewer.cam.azimuth = 135  # degrees, horizontal rotation
    self.viewer.cam.elevation = -20  # degrees, negative looks down
    self.viewer.cam.distance = 2.5  # meters from lookat point
    self.viewer.cam.lookat[:] = list(
      self.default_base[0:3]
    )  # (x, y, z) point to look at

    self.viewer_render_hz = 50.0
    self._last_viewer_sync = 0.0
    self._real_start_time = time.perf_counter()
    self._next_step_deadline = self._real_start_time + self.sim_dt

  # anchor goal targets to the robot's initial pelvis pose in world frame
  def init_goals(self):
    goals_cfg = self.config.get("goals", [])

    # populate body transforms from the initial qpos set in init_simulation()
    mujoco.mj_kinematics(self.mj_model, self.mj_data)
    pelvis_pos = self.mj_data.body("pelvis").xpos.astype(np.float32)
    pelvis_quat = self.mj_data.body("pelvis").xquat.astype(np.float32)  # [w, x, y, z]
    R_init = quat_to_rotation_matrix(pelvis_quat)

    self._goal_types: list[str] = []
    self._goal_pos_w: list[np.ndarray] = []  # world-frame position targets
    self._goal_vel_w: list[np.ndarray] = []  # world-frame velocity targets
    self._goal_quat_w: list[np.ndarray] = []  # world-frame quaternion [w,x,y,z] targets
    for goal in [goal for goal in goals_cfg if goal["motion_index"] == self.motion_idx]:
      vec = np.array(goal["vector"], dtype=np.float32)
      goal_type = goal["type"]
      self._goal_types.append(goal_type)
      if goal_type == "position":
        # rotate body-frame offset into world frame and translate by pelvis origin
        self._goal_pos_w.append(R_init @ vec + pelvis_pos)
        self._goal_vel_w.append(np.zeros(3, dtype=np.float32))
        self._goal_quat_w.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
      elif goal_type == "velocity":
        self._goal_pos_w.append(np.zeros(3, dtype=np.float32))
        # velocity vec is in body frame at init; rotate into world frame
        self._goal_vel_w.append(vec)
        self._goal_quat_w.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
      elif goal_type == "orientation":
        self._goal_pos_w.append(np.zeros(3, dtype=np.float32))
        self._goal_vel_w.append(np.zeros(3, dtype=np.float32))
        # vec is RPY offset relative to initial anchor; anchor into world frame
        # to match training: target_orientation_w = anchor_quat * rpy_to_quat(rpy)
        self._goal_quat_w.append(quat_multiply(pelvis_quat, rpy_to_quat(vec)))
      else:
        raise ValueError(f"Unsupported goal type: {goal_type!r}")

    motion_goal_names = [
      g["name"] for g in goals_cfg if g["motion_index"] == self.motion_idx
    ]
    print(f"Goals initialized for motion {self.motion_idx}: {motion_goal_names}")

  #################################################################
  # PUBLISHING AND CALLBACKS
  #################################################################

  # command callback: [q_des, dq_des, Kp, Kd, tau_ff] (nu * 5)
  def command_callback(self, msg):
    data = np.array(msg.data)

    # unpack the command (same order as hardware)
    self.command_received = True
    self.qpos_des = data[0 * self.nu : 1 * self.nu]
    self.qvel_des = data[1 * self.nu : 2 * self.nu]
    self.Kp = data[2 * self.nu : 3 * self.nu]
    self.Kd = data[3 * self.nu : 4 * self.nu]
    self.tau_ff = data[4 * self.nu : 5 * self.nu]

  # motion frame callback
  def motion_frame_callback(self, msg):
    self.motion_frame = int(msg.data)

  # which motion callback
  def which_motion_callback(self, msg):
    new_idx = 1 if msg.data[2] > 0.0 else 0
    if new_idx != self.motion_idx:
      self.motion_idx = new_idx
      self.init_goals()

  # publish pelvis IMU: [rpy(3), quat(4), gyro(3), acc(3)]
  def publish_pelvis_imu(self):
    pelvis_quat = self.mj_data.sensor("pelvis_imu_quat_sensor").data.copy()
    pelvis_gyro = self.mj_data.sensor("pelvis_imu_gyro_sensor").data.copy()
    pelvis_acc = self.mj_data.sensor("pelvis_imu_acc_sensor").data.copy()
    pelvis_rpy = quat_to_rpy(pelvis_quat)

    pelvis_msg = Float32MultiArray()
    pelvis_msg.data = np.concatenate(
      [pelvis_rpy, pelvis_quat, pelvis_gyro, pelvis_acc]
    ).tolist()
    self.pelvis_imu_state_pub.publish(pelvis_msg)

  # publish joint state: [q(nu), dq(nu), ddq(nu), tau_est(nu)]
  def publish_joint_state(self):
    qpos_joints = np.array(
      [self.mj_data.sensor(name).data[0] for name in self.joint_pos_sensor_names]
    )
    qvel_joints = np.array(
      [self.mj_data.sensor(name).data[0] for name in self.joint_vel_sensor_names]
    )
    ddq_joints = np.zeros(
      self.nu
    )  # NOTE: no such thing as joint acceleration sensors in Mujoco, so we publish zeros here
    tau_est_joints = self.mj_data.ctrl[: self.nu].copy()

    joint_state_msg = Float32MultiArray()
    joint_state_msg.data = np.concatenate(
      [qpos_joints, qvel_joints, ddq_joints, tau_est_joints]
    ).tolist()

    self.joint_state_pub.publish(joint_state_msg)

  # publish goal positions, orientations, and velocities according to the yaml config
  def publish_goals(self):
    which_motion_msg = Float64()
    which_motion_msg.data = float(self.motion_idx)
    self.which_motion_pub.publish(which_motion_msg)

    if not self._goal_types:
      return

    pelvis_pos = self.mj_data.body("pelvis").xpos.astype(np.float32)
    pelvis_quat = self.mj_data.body("pelvis").xquat.astype(np.float32)  # [w, x, y, z]
    R = quat_to_rotation_matrix(pelvis_quat)

    pelvis_quat_inv = quat_conjugate(pelvis_quat)

    goal_vecs = []
    for goal_type, pos_w, vel_w, quat_w in zip(
      self._goal_types, self._goal_pos_w, self._goal_vel_w, self._goal_quat_w
    ):
      if goal_type == "position":
        # position in anchor (pelvis) frame, matching quat_apply(quat_inv(anchor), pos_w - anchor_pos)
        goal_vecs.append(R.T @ (pos_w - pelvis_pos))
      elif goal_type == "velocity":
        # velocity is kept in world frame in training (no transform applied)
        goal_vecs.append(vel_w)
      elif goal_type == "orientation":
        # 4-value quaternion in anchor frame: quat_inv(anchor) * target_ori_w
        goal_vecs.append(quat_multiply(pelvis_quat_inv, quat_w))
    goal_msg = Float32MultiArray()
    goal_msg.data = np.concatenate(goal_vecs).tolist()
    self.goal_pub.publish(goal_msg)

  #################################################################
  # SIMULATION
  #################################################################

  # add per-sensor Gaussian noise in place on mj_data.sensordata using the
  # std devs declared in the XML (sensor_noise[i] is the std dev for sensor i)
  def _apply_sensor_noise(self):
    for i in range(self.mj_model.nsensor):
      std = self.mj_model.sensor_noise[i]
      if std <= 0.0:
        continue
      adr = self.mj_model.sensor_adr[i]
      dim = self.mj_model.sensor_dim[i]
      self.mj_data.sensordata[adr : adr + dim] += np.random.normal(0.0, std, size=dim)

  # compute torque using PD control + feedforward
  def compute_torque(self):
    # get current joint positions and velocities
    qpos_joints = self.mj_data.qpos[7 : 7 + self.nu]
    qvel_joints = self.mj_data.qvel[6 : 6 + self.nu]

    # tau = kp * (qpos_des - qpos) + kd * (qvel_des - qvel) + tau_ff
    tau = (
      self.Kp * (self.qpos_des - qpos_joints)
      + self.Kd * (self.qvel_des - qvel_joints)
      + self.tau_ff
    )

    return tau

  # step the simulation
  def step_simulation(self):
    # compute the torque to apply
    if self.command_received == True:
      tau = self.compute_torque()
      self.mj_data.ctrl[:] = tau
    else:
      self.mj_data.ctrl[:] = 0.0

    # step the simulation
    mujoco.mj_step(self.mj_model, self.mj_data)

    # inject sensor noise (newer mujoco dropped the sensornoise, so we
    # apply the per-sensor std devs from XML noise="..." attrs by hand)
    if self.apply_noise:
      self._apply_sensor_noise()

    # publish simulation time
    time_msg = Float64()
    time_msg.data = self.mj_data.time
    self.simulation_time_pub.publish(time_msg)

    # sync viewer at viewer_render_hz
    now = time.perf_counter()
    if (
      self.viewer.is_running()
      and (now - self._last_viewer_sync) >= 1.0 / self.viewer_render_hz
    ):
      # update the viewer with the current simulation state
      self.viewer.sync()

      # display sim time first and wall-clock elapsed time second
      real_elapsed = now - self._real_start_time
      self.viewer.set_texts(
        (
          self._viewer_font_scale,
          mujoco.mjtGridPos.mjGRID_TOPLEFT,
          f"Sim time:   {self.mj_data.time:.2f}s\nReal time: {real_elapsed:.2f}s",
          "",
        )
      )

      self._last_viewer_sync = now

    # Real-time sync against an absolute schedule so drift does not accumulate.
    remaining = self._next_step_deadline - time.perf_counter()
    if remaining > 0.0:
      time.sleep(remaining)

    # Keep a fixed absolute schedule so the loop can catch up after overruns.
    self._next_step_deadline += self.sim_dt

  # shutdown the node and close the viewer
  def destroy_node(self):
    if self.viewer.is_running():
      self.viewer.close()
    super().destroy_node()


############################################################################
# MAIN FUNCTION
############################################################################


def main(args=None):
  # init ROS2
  rclpy.init()

  # parse arguments
  parser = argparse.ArgumentParser(
    description="Asynchronous Simulation Node using Mujoco."
  )
  # config path argument
  parser.add_argument(
    "--config",
    type=str,
    required=True,
    help='Path to the config yaml file. Example: "g1_29dof.yaml".',
  )
  # enable sensor noise (off by default)
  parser.add_argument(
    "--noise",
    action="store_true",
    help="Enable per-sensor Gaussian noise injection. Noise is off by default.",
  )
  args = parser.parse_args()

  # create the simulation node
  sim_node = SimulationNode(args.config, apply_noise=args.noise)

  # run normally
  try:
    while rclpy.ok() and sim_node.viewer.is_running():
      rclpy.spin_once(sim_node, timeout_sec=0.1)
  except KeyboardInterrupt:
    pass
  # ROS2 shutdown
  finally:
    sim_node.destroy_node()
    if rclpy.ok():
      rclpy.shutdown()

  print("Simulation shutdown complete.")


if __name__ == "__main__":
  main()
