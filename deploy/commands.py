##
#
# Run joystick commands
#
##

# standard imports
import argparse
import subprocess
from dataclasses import dataclass

# other imports
import time
import yaml

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray

# directory imports
import os
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")


############################################################################
# COMMAND NODE
############################################################################

# Xbox buttons
@dataclass
class JoystickState:
    
    # Buttons
    A: int = 0
    B: int = 0
    X: int = 0
    Y: int = 0
    LB: int = 0
    RB: int = 0
    LMB: int = 0   # left middle button
    RMB: int = 0   # right middle button
    XBOX: int = 0  # middle button with Xbox logo
    LS: int = 0    # left stick press
    RS: int = 0    # right stick press
    MB: int = 0    # middle button between LMB and RMB

    # Axes
    LX: float = 0.0
    LY: float = 0.0
    RX: float = 0.0
    RY: float = 0.0
    LT: float = 0.0
    RT: float = 0.0
    L_DPAD: float = 0.0  # D-PAD left
    R_DPAD: float = 0.0  # D-PAD right
    U_DPAD: float = 0.0  # D-PAD up
    D_DPAD: float = 0.0  # D-PAD down

class CommandsNode(Node):
    """
    Use joystick to publish commands to the simulation and control nodes.
    """

    def __init__(self, config_path: str = None):

        super().__init__('command_node')

        # load config file
        self.config = self.load_config(config_path)

        # launch joy_node as a subprocess
        self.joy_process = subprocess.Popen(
            ['ros2', 'run', 'joy', 'joy_node'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        print("Launched joy_node as a subprocess.")

        # joystick state
        self.joystick_state = JoystickState()

        # ROS2 subscribers
        self.joy_sub = self.create_subscription(Joy, 'joy', self.joy_callback, 10)
        self.joy_msg = None
        
        # wait for the first joystick message to be received
        t0 = time.time()
        print("Waiting for the first joystick message...")
        while self.joy_msg is None:
            rclpy.spin_once(self, timeout_sec=0.01)
        print("Received the first joystick message.")

        # ROS2 publishers
        self.command_pub = self.create_publisher(Float32MultiArray, 'command', 10)

        # create timer to publish commands at a fixed rate
        command_freq = 100.0
        self.command_timer = self.create_timer(1.0 / command_freq, self.publish_command)

        print("Command node initialized.")

    # load the config file
    def load_config(self, config_path: str):
        # open the config file and load it
        config_path_full = ROOT_DIR + "/deploy/configs/" + config_path
        with open(config_path_full, 'r') as f:
            config = yaml.safe_load(f)

        print(f"Loaded config from [{config_path_full}].")

        return config

    # callback for joystick messages
    def joy_callback(self, msg: Joy):
        self.joy_msg = msg

    # parse the joystick message
    def joy_parse(self):
        # buttons
        self.joystick_state.A = self.joy_msg.buttons[0]
        self.joystick_state.B = self.joy_msg.buttons[1]
        self.joystick_state.X = self.joy_msg.buttons[2]
        self.joystick_state.Y = self.joy_msg.buttons[3]
        self.joystick_state.LB = self.joy_msg.buttons[4]
        self.joystick_state.RB = self.joy_msg.buttons[5]
        self.joystick_state.LMB = self.joy_msg.buttons[6]
        self.joystick_state.RMB = self.joy_msg.buttons[7]
        self.joystick_state.XBOX = self.joy_msg.buttons[8]
        self.joystick_state.LS = self.joy_msg.buttons[9]
        self.joystick_state.RS = self.joy_msg.buttons[10]
        self.joystick_state.MB = self.joy_msg.buttons[11]

        # joysticks
        self.joystick_state.LX = self.joy_msg.axes[0]
        self.joystick_state.LY = self.joy_msg.axes[1]
        self.joystick_state.RX = self.joy_msg.axes[3]
        self.joystick_state.RY = self.joy_msg.axes[4]

        # triggers go from 1 (not pressed) to -1 (fully pressed), convert to 0 to 1
        self.joystick_state.LT = -0.5 * self.joy_msg.axes[2] + 0.5
        self.joystick_state.RT = -0.5 * self.joy_msg.axes[5] + 0.5

        # D-PAD axes
        DPAD_horizontal = self.joy_msg.axes[6]
        if DPAD_horizontal == 1.0:
            self.joystick_state.L_DPAD = 1.0
            self.joystick_state.R_DPAD = 0.0
        elif DPAD_horizontal == -1.0:
            self.joystick_state.L_DPAD = 0.0
            self.joystick_state.R_DPAD = 1.0
        else:
            self.joystick_state.L_DPAD = 0.0
            self.joystick_state.R_DPAD = 0.0

        DPAD_vertical = self.joy_msg.axes[7]
        if DPAD_vertical == -1.0:
            self.joystick_state.U_DPAD = 0.0
            self.joystick_state.D_DPAD = 1.0
        elif DPAD_vertical == 1.0:
            self.joystick_state.U_DPAD = 1.0
            self.joystick_state.D_DPAD = 0.0
        else:
            self.joystick_state.U_DPAD = 0.0
            self.joystick_state.D_DPAD = 0.0

    # publish command based on the latest joystick message
    def publish_command(self):
        # update the joystick
        self.joy_parse()

        # convert the joystick state to a command message
        vx_cmd = self.joystick_state.LY
        vy_cmd = self.joystick_state.LX
        omega_cmd = self.joystick_state.RX

        # publish the command
        cmd_msg = Float32MultiArray()
        cmd_msg.data = [vx_cmd, vy_cmd, omega_cmd]

        self.command_pub.publish(cmd_msg)

    # terminate joy_node on shutdown
    def destroy_node(self):
        if self.joy_process is not None:
            self.joy_process.terminate()
            self.joy_process.wait()
        super().destroy_node()


############################################################################
# MAIN FUNCTION
############################################################################

def main():

    # init ROS2
    rclpy.init()

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Asynchronous Simulation Node using Mujoco.'
    )
    # config path argument
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the Mujoco config yaml file. Example: "g1_29dof.yaml".'
    )
    args = parser.parse_args()

    # create command node
    cmd_node = CommandsNode(args.config)

    # execute the simulation
    try:
        # spin the node
        rclpy.spin(cmd_node)

    except KeyboardInterrupt:
        pass

    finally:
        # close everything
        cmd_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()