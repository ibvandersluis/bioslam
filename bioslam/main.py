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
X_OUT = 20 # X size of output layer
Y_OUT = 20 # Y size of output layer
SIGMA0 = max(X_OUT, Y_OUT)/2 # Radius of map at t0
LAM = 500 # Lambda, the time scaling constant
L0 = 0.3 # Initial learning rate

class Listener(BaseListener):

    def __init__(self):
        super().__init__('bioslam')

        self.t = 0 # Timestep (iteration count)

        # Initialise arrays
        self.coords = np.indices((X_OUT, Y_OUT))
        self.coords = np.flip(self.coords.T, axis=2).reshape((X_OUT * Y_OUT, 2))
        self.diff = None # Difference between inputs and weights (capture - W)
        self.bmu_dists = []

        # Initialize weights
        self.w = np.random.rand(X_OUT, Y_OUT, IN_MAX, IN_VECT)
        norm = np.linalg.norm(self.w)
        self.w /= norm # Normalise (? ask Alex)

        # Set publishers
        self.map_pub = self.create_publisher(ConeArray, '/mapping/map', 10)
        self.pose_pub = self.create_publisher(CarPos, '/mapping/position', 10)

        # Set subscribers
        self.cones_sub = self.create_subscription(ConeArray, '/cones/positions', self.cones_callback, 10)
    
    def compute_bmu_distances(self, bmu):
        diff = bmu - self.coords
        d_sq = diff[:, 0]**2 + diff[:, 1]**2
        d = np.sqrt(d_sq).reshape((X_OUT, Y_OUT))

        return d

    def cones_callback(self, msg: ConeArray):
        t = self.t
        if (self.sigma(t) < 1):
            return
        # Place x y positions of cones into input array x
        x = np.array([[cone.x, cone.y] for cone in msg.cones])
        # Pad capture array with zeros for give it a shape of (IN_MAX, IN_VECT)
        x = np.vstack((x, np.zeros((30 - len(x[:, 0]), IN_VECT))))
        a = time.time()
        bmu = self.get_bmu(x) # Determine BMU coordinates
        b = time.time()
        dist = self.compute_bmu_distances(bmu) # Calculate distance from each node to BMU
        c = time.time()
        self.update(dist, t) # Compute weights of neighbourhood nodes
        d = time.time()
        print('get bmu: ' + str(b-a))
        print('compute dist: ' + str(c-b))
        print('compute weight: ' + str(d-c))
        print('t: ' + str(t), '\nlambda: ' + str(self.L(t)), '\nrad: ' + str(self.sigma(t)))
        self.t += 1 # Increment timestep
        print(bmu)
        print('Quantisation error: ' + str(self.quant_err()))
        # self.plot_som()

    def get_bmu(self, x):
        """
        Determines the X-Y position of the winning node in the output layer
        """
        # Make self.capture 4D
        j = time.time()
        cap = np.zeros((1, 1, IN_MAX, IN_VECT))
        cap += x
        k = time.time()
        self.diff = cap - self.w # Calculate differences
        l = time.time()
        # Calculate distances between the input and each node
        results = np.sqrt(np.sum((self.w - cap)**2, axis=(2, 3)))
        m = time.time()
        # Get X-Y position of the node with the minimum distance
        bmu = np.unravel_index(np.argmin(results, axis = None), results.shape)
        n = time.time()
        self.bmu_dists.append(np.linalg.norm(self.diff[bmu]))
        print('make 4d: ' + str(k-j))
        print('diff: ' + str(l-k))
        print('results: ' + str(m-l))
        print('bmu: ' + str(n-m))

        return bmu

    def get_x_y(self):
        x = self.w[:, :, :, 0].flatten()
        y = self.w[:, :, :, 1].flatten()

        return x, y

    def L(self, t):
        return L0 * np.exp(-t/LAM)

    def N(self, dist, t):
        """
        Computes the neighbouring penalty N

        :param dist: An array of distances to the BMU, matching the shape of the output layer
        :param t: The current timestep
        """
        sigma_t = self.sigma(t)
        return np.exp(-(dist**2)/(2 * sigma_t**2))

    def plot_som(self):
        x, y = self.get_x_y()
        plt.scatter(x, y, marker='.', c='k')
        plt.show()
        plt.pause(0.0001)
        plt.clf()
    
    def quant_err(self):
        return np.array(self.bmu_dists).mean()
    
    def sigma(self, t):
        return SIGMA0 * np.exp(-t/LAM)

    def update(self, dist, t):
        Ndt = self.N(dist, t).reshape((X_OUT, Y_OUT, 1, 1))
        self.w += Ndt * self.L(t) * self.diff

def main(args=None):
    rclpy.init(args=args)

    node = Listener()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv)
