##
#
# Control node for the MuJoCo simulation.
#
##

# standard imports
import time

# other imports
import numpy as np
import yaml

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float64, Float32MultiArray