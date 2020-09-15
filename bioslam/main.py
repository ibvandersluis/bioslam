#!/usr/bin/env python3

import sys

import message_filters
import rclpy
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import scipy.stats
from copy import deepcopy
from helpers.listener import BaseListener
from helpers import shortcuts
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from obr_msgs.msg import Cone, CarPos, ConeArray, IMU, Label
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Point, Twist, Vector3
from gazebo_msgs.msg import LinkStates
from std_msgs.msg import String

# VERSION 1

# Input: 2 input vectors, 1 for each of the x and y position values at a given timestep
# Output: A 2x2 grid of nodes, each represented by a set of weights connected to each input

# Declare constants
IN_MAX = 30 # Max input amount (i.e. max number of cones at once)
IN_VECT = 2 # The number of input vectors
X_OUT = 500 # X size of output layer
Y_OUT = 500 # Y size of output layer
MAP_R = max(X_OUT, Y_OUT)/2 # Radius of map at t0
ITER = 50000 # Number of iterations
TIME_CONST = ITER / np.log(MAP_R)
LAM = 0.3    # λ coefficient
DLAM = 0.05  # Δλ

def distance(w, x):
    r = 0
    for i in range(len(w)):
        r = r + (w[i] - x[i])*(w[i] - x[i])
    
    r = np.sqrt(r)
    return r

class Listener(BaseListener):

    def __init__(self):
        super().__init__('bioslam')

        # Initialise some member variables
        self.t = 0 # Timestep (iteration count)
        self.lam = LAM # The learning rate, λ coefficient
        self.n_rad = MAP_R # Neighbourhood radius

        # Initialize weights
        self.w = np.random.rand(X_OUT, Y_OUT, IN_MAX, IN_VECT)
        norm = np.linalg.norm(self.w)
        self.w /= norm # Normalise (? ask Alex)
        self.diff = None # Difference between inputs and weights (capture - W)
        self.distances = None

        # Initialize inputs
        self.capture = []
        self.X = []

        # Set publishers
        self.map_pub = self.create_publisher(ConeArray, '/mapping/map', 10)
        self.pose_pub = self.create_publisher(CarPos, '/mapping/position', 10)
        self.cmd_pub = self.create_publisher(Twist, '/gazebo/cmd_vel', 10)

        # Set subscribers
        self.cones_sub = self.create_subscription(ConeArray, '/cones/positions', self.cones_callback, 10)
        self.gnss_sub = self.create_subscription(NavSatFix, '/peak_gps/gps', self.gnss_callback, 10)
        self.imu_sub = self.create_subscription(IMU, '/peak_gps/imu', self.imu_callback, 10)
        self.control_sub = self.create_subscription(Twist, '/gazebo/cmd_vel', self.control_callback, 10)
    
    def compute_influence(self, x, y):
        theta = np.exp(-(self.distances[x, y]**2)/(2 * self.n_rad**2))

        return theta

    def compute_lambda(self):
        self.lam = LAM * np.exp(-self.t/ITER)

    def compute_radius(self):
        self.n_rad = MAP_R * np.exp(-self.t/TIME_CONST)

    def compute_weights(self, x, y):
        old_weights = self.w[x, y]
        theta = self.compute_influence(x, y)
        new_weights = old_weights + theta * self.lam * self.diff[x, y]

        return new_weights

    def cones_callback(self, msg: ConeArray):
        # Place x y positions of cones into self.capture
        self.capture = np.array([[cone.x, cone.y] for cone in msg.cones])
        # Pad capture array with zeros for give it a shape of (IN_MAX, IN_VECT)
        self.capture = np.vstack((np.capture, np.zeros((30 - len(self.capture[:, 0]), IN_VECT))))
        print(self.capture)

        bmu = self.get_bmu()

        neighbourhood = np.argwhere(self.distances < self.n_rad)

    def control_callback(self, msg: Twist):
        str(msg) # For some reason this is needed to access msg.linear.x
        self.v = msg.linear.x
        self.yaw = msg.angular.z
        self.u = np.array([self.v, self.yaw]).reshape(2, 1)

        self.get_logger().info(f'Command confirmed: {msg.linear.x} m/s turning at {msg.angular.z} rad/s')

    def get_bmu(self):
        """
        Determines the X-Y position of the winning node in the output layer
        """
        # Make self.capture 4D
        cap = np.zeros((1, 1, IN_MAX, IN_VECT))
        cap += self.capture

        self.diff = cap - self.w # Calculate differences

        # Calculate distances between the input and each node
        results = np.sqrt(np.sum((self.w - cap)**2, axis=(2, 3)))
        # Get X-Y position of the node with the minimum distance
        bmu = np.unravel_index(np.argmin(results, axis = None), results.shape)
        self.distances = np.copy(results)

        return bmu

    def gnss_callback(self, msg: NavSatFix()):
        # Log data retrieval
        self.get_logger().info(f'From GNSS: {msg.latitude}, {msg.longitude}')
    
    def imu_callback(self, msg: IMU()):
        # Log data retrieval
        self.get_logger().info(f'From IMU: {msg.longitudinal}, {msg.lateral}, {msg.vertical}')

    def link_states_callback(self, links_msg: LinkStates):
        cones = []
        for name, pose in zip(links_msg.name, links_msg.pose):
            if 'blue_cone' in name:
                label = Label.BLUE_CONE
            elif 'yellow_cone' in name:
                label = Label.YELLOW_CONE
            elif 'big_orange_cone' in name:
                label = Label.BIG_ORANGE_CONE
            elif 'orange_cone' in name:
                label = Label.ORANGE_CONE
            else:
                # if not a cone
                continue
            cones.append(Cone(position=pose.position, label=Label(label=label)))

    def timer_callback(self):
        print('timer_callback()')

def main(args=None):
    rclpy.init(args=args)

    node = Listener()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv)
