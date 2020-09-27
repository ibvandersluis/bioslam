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

# Input: 2 input vectors, 1 for each of the x and y position values at a given timestep
# Output: A 2x2 grid of nodes, each represented by a set of weights connected to each input

# Declare constants
IN_MAX = 30 # Max input amount (i.e. max number of cones at once)
IN_VECT = 2 # The number of input vectors
X_OUT = 40 # X size of output layer
Y_OUT = 40 # Y size of output layer
SIGMA0 = max(X_OUT, Y_OUT)/2 # Radius of map at t0
LAM = 5000 # Lambda, the time scaling constant
L0 = 0.02 # Initial learning rate

class Listener(BaseListener):

    def __init__(self):
        super().__init__('bioslam')

        self.t = 0 # Timestep (iteration count)

        # Initialise arrays
        self.coords = np.indices((X_OUT, Y_OUT)) # An array of all coordinates for the output layer
        self.coords = np.flip(self.coords.T, axis=2).reshape((X_OUT * Y_OUT, 2))
        self.diff = None # Difference between inputs and weights (x - w)
        self.bmu_dists = []

        # Initialize weights
        # First layer takes observed cones as input
        self.w = np.random.rand(X_OUT, Y_OUT, IN_MAX, IN_VECT)
        # Second layer takes BMU of first layer as input
        self.w2 = np.random.rand(X_OUT, Y_OUT, IN_VECT)

        # Set publishers
        self.map_pub = self.create_publisher(ConeArray, '/mapping/map', 10)
        self.pose_pub = self.create_publisher(CarPos, '/mapping/position', 10)

        # Set subscribers
        self.cones_sub = self.create_subscription(ConeArray, '/cones/positions', self.cones_callback, 10)
    
    def compute_bmu_distances(self, bmu):
        """
        Calculates the Euclidian distance between all output nodes and the BMU

        :param bmu: The coordinates of the BMU on the output layer
        :return: d, a matrix of distances corresponding to node coordinates
        """
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
        # Pad capture array with zeros for give it a shape of (1, 1, IN_MAX, IN_VECT)
        x = np.vstack((x, np.zeros((30 - len(x[:, 0]), IN_VECT)))).reshape((1, 1, IN_MAX, IN_VECT))
        
        bmu = self.get_bmu(x, self.w) # Determine BMU coordinates
        
        dist = self.compute_bmu_distances(bmu) # Calculate distance from each node to BMU
        
        self.update(dist, t) # Compute weights of neighbourhood nodes
        
        self.t += 1 # Increment timestep
        print(bmu)
        print('Quantisation error: ' + str(self.quant_err()))
        self.plot_bmu(bmu)

    def get_bmu(self, x, w):
        """
        Determines the X-Y position of the winning node in the first layer

        :param x: The input array. Must match dimensionality of w array
        :param w: The weights
        :return: bmu, the coordinates of the best matching unit
        """        
        self.diff = x - w # Calculate differences
        
        # Calculate distances between the input and each node
        if (x.ndim == 4):
            results = np.sqrt(np.sum((w - x)**2, axis=(2, 3)))
        elif (x.ndim == 3):
            results = np.sqrt(np.sum((w - x)**2, axis=2))
        
        # Get X-Y position of the node with the minimum distance
        bmu = np.unravel_index(np.argmin(results, axis = None), results.shape)
        
        if (x.ndim == 3):
            self.bmu_dists.append(np.linalg.norm(self.diff[bmu]))

        return bmu

    def get_x_y(self):
        """
        Collects x and y weights from the output layer for plotting
        # Currently doesn't work
        """
        x = self.w[:, :, :, 0].flatten()
        y = self.w[:, :, :, 1].flatten()

        return x, y

    def L(self, t):
        """
        Calculates the learning rate for the given timestep

        :param t: The given timestep
        :return: Lt
        """
        return L0 * np.exp(-t/LAM)

    def N(self, dist, t):
        """
        Computes the neighbouring penalty N

        :param dist: An array of distances to the BMU, matching the shape of the output layer
        :param t: The given timestep
        """
        sigma_t = self.sigma(t)
        return np.exp(-(dist**2)/(2 * sigma_t**2))

    def plot_bmu(self, bmu):
        """
        Plots the coordinates of the BMU

        :param bmu: X-Y coordinates of the best matching unit
        """
        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event', lambda event:
            [exit(0) if event.key == 'escape' else None])
        plt.plot(bmu[0], bmu[1], "*r", label='BMU')
        plt.legend()
        plt.title('BioSLAM: BMU')
        plt.xlabel('X coord in output layer')
        plt.ylabel('Y coord in output layer')
        plt.xlim(0, X_OUT)
        plt.ylim(0, Y_OUT)
        plt.grid(True)
        plt.pause(0.001)

    def plot_som(self):
        """
        Plots the SOM by taking x, y weights from the output layer
        """
        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event', lambda event:
            [exit(0) if event.key == 'escape' else None])
        x, y = self.get_x_y()
        plt.plot(x, y, ".k", label='nodes')
        plt.legend()
        plt.title('BioSLAM: SOM')
        plt.xlabel('X Weights')
        plt.ylabel('Y Weights')
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)

    
    def quant_err(self):
        """
        Measures the quantisation error, the average difference between the input and the BMU

        :return: The quantisation error
        """
        return np.array(self.bmu_dists).mean()
    
    def sigma(self, t):
        """
        Calculates sigma(t), the neighbourhood radius for the given timestep

        :param t: The given timestep
        :return: sigma(t)
        """
        return SIGMA0 * np.exp(-t/LAM)

    def update(self, dist, t):
        """
        Calculates new weights for the SOM

        :param dist: The array of distances for each node from the BMU
        :param t: The given timestep
        """
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
