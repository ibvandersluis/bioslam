#!/usr/bin/env python3

import sys

import message_filters
import rclpy
import numpy as np
import matplotlib.pyplot as plt
import time
from helpers.listener import BaseListener
from helpers import shortcuts
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from obr_msgs.msg import Cone, CarPos, ConeArray, IMU, Label
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Point, Twist, Vector3
from gazebo_msgs.msg import LinkStates

# VERSION 1

# Input: 2 input vectors, 1 for each of the x and y position values at a given timestep
# Output: A 2x2 grid of nodes, each represented by a set of weights connected to each input

# Declare constants
IN_MAX = 30 # Max input amount (i.e. max number of cones at once)
IN_VECT = 2 # The number of input vectors
X_OUT = 100 # X size of output layer
Y_OUT = 100 # Y size of output layer
MAP_R = max(X_OUT, Y_OUT)/2 # Radius of map at t0
ITER = 1000 # Number of iterations
TIME_CONST = ITER / np.log(MAP_R)
LAM = 0.3    # λ coefficient
DLAM = 0.05  # Δλ

class Listener(BaseListener):

    def __init__(self):
        super().__init__('bioslam')

        # Initialise some member variables
        self.t = 0 # Timestep (iteration count)
        self.lam = LAM # The learning rate, λ coefficient
        self.n_rad = MAP_R # Neighbourhood radius

        # Initialise arrays
        self.coords = np.indices((X_OUT, Y_OUT))
        self.coords = np.flip(self.coords.T, axis=2).reshape((X_OUT * Y_OUT, 2))
        self.diff = None # Difference between inputs and weights (capture - W)
        self.distances = None # Array of distances to the BMU

        # Initialize weights
        self.w = np.random.rand(X_OUT, Y_OUT, IN_MAX, IN_VECT)
        norm = np.linalg.norm(self.w)
        self.w /= norm # Normalise (? ask Alex)

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
    
    def compute_bmu_distances(self, bmu):
        diff = bmu - self.coords
        d_sq = diff[:, 0]**2 + diff[:, 1]**2
        d = np.sqrt(d_sq).reshape((X_OUT, Y_OUT))

        self.distances = d

    def compute_influence(self):
        theta = np.exp(-(self.distances**2)/(2 * self.n_rad**2))

        return theta

    def compute_lambda(self):
        self.lam = LAM * np.exp(-self.t/ITER)

    def compute_radius(self):
        self.n_rad = MAP_R * np.exp(-self.t/TIME_CONST)

    def compute_weights(self, nbh):
        old_weights = self.w[nbh[:, 0], nbh[:, 1]]
        theta = self.compute_influence()[nbh[:, 0], nbh[:, 1]].reshape(len(old_weights[:]), 1, 1)
        self.w[nbh[:, 0], nbh[:, 1]] = old_weights + theta * self.lam * self.diff[nbh[:, 0], nbh[:, 1]]

    def cones_callback(self, msg: ConeArray):
        if (self.n_rad < 1):
            return
        # Place x y positions of cones into self.capture
        self.capture = np.array([[cone.x, cone.y] for cone in msg.cones])
        # Pad capture array with zeros for give it a shape of (IN_MAX, IN_VECT)
        self.capture = np.vstack((self.capture, np.zeros((30 - len(self.capture[:, 0]), IN_VECT))))
        a = time.time()
        bmu = self.get_bmu() # Determine BMU coordinates
        b = time.time()
        self.compute_bmu_distances(bmu) # Calculate distance from each node to BMU
        c = time.time()
        nbh = np.argwhere(self.distances <= self.n_rad) # Get indices of neighbourhood
        d = time.time()
        self.compute_weights(nbh) # Compute weights of neighbourhood nodes
        e = time.time()
        print('get bmu: ' + str(b-a))
        print('compute dist: ' + str(c-b))
        print('get nbh: ' + str(d-c))
        print('compute weight: ' + str(e-d))
        print('t: ' + str(self.t), '\nlambda: ' + str(self.lam), '\nrad: ' + str(self.n_rad))
        self.t += 1 # Increment timestep
        self.compute_lambda()
        self.compute_radius()
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
        j = time.time()
        cap = np.zeros((1, 1, IN_MAX, IN_VECT))
        cap += self.capture
        k = time.time()
        self.diff = cap - self.w # Calculate differences
        l = time.time()
        # Calculate distances between the input and each node
        results = np.sqrt(np.sum((self.w - cap)**2, axis=(2, 3)))
        m = time.time()
        # Get X-Y position of the node with the minimum distance
        bmu = np.unravel_index(np.argmin(results, axis = None), results.shape)
        n = time.time()
        print('make 4d: ' + str(k-j))
        print('diff: ' + str(l-k))
        print('results: ' + str(m-l))
        print('bmu: ' + str(n-m))

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
