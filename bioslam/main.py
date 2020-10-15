#!/usr/bin/env python3

import os
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

np.set_printoptions(threshold=sys.maxsize)

# Input: 2 input vectors
#   1 for each of the x and y position values at a given timestep
# First layer: A (X_OUT x Y_OUT) point latice
#   each represented by a set of weights connected to each input. Outputs BMU
# Output layer: A (X_OUT x Y_OUT) point latice taking 1st-layer BMU as input
#   each unit has weights [x, y] to match the values of the input

# Declare constants
IN_MAX = 30 # Max input amount (i.e. max number of cones at once)
IN_VECT = 2 # The number of input vectors
X_OUT = 40 # X size of output layer
Y_OUT = 40 # Y size of output layer
THRESHOLD = 1.0 # Threshold for Sigma to stop training
SIGMA0 = max(X_OUT, Y_OUT)/2 # Radius of map at t0
LAM = 140 # Lambda, the time scaling constant
L0 = 0.3 # Initial learning rate
PLOTTING = True
DEBUGGING = False

def L(t):
    """
    Calculates the learning rate for the given timestep. The radius decays
    gradually as the network continues to train.

    :param t: The given timestep
    :return: L(t)
    """
    return L0 * np.exp(-t/LAM)

def sigma(t):
    """
    Calculates sigma(t), the neighbourhood radius for the given timestep. The
    radius decays gradually as the network continues to train.

    :param t: The given timestep
    :return: sigma(t)
    """
    return SIGMA0 * np.exp(-t/LAM)

def theta(dist, t):
    """
    Computes the neighbouring penalty theta. This gaussian decay function
    causes unite near the BMU to be updated strongly, while units near the
    edge of the neighbourhood are hardly changed.

    :param dist: An array of distances to the BMU, matching output layer shape
    :param t: The given timestep
    :return: theta(d, t)
    """
    # sigma_t = sigma(t)
    return np.exp(-(dist**2)/(2 * sigma(t)**2))

class Listener(BaseListener):

    def __init__(self):
        super().__init__('bioslam')

        if(DEBUGGING):
            self.debug = 0
            self.path = os.getcwd() + '/bioslam_debug'
            try:
                os.mkdir(self.path)
            except:
                pass

        self.t = 0 # Timestep (iteration count)

        # Initialise arrays
        self.coords = np.indices((X_OUT, Y_OUT)) # An array of all unit coords
        self.coords = np.flip(self.coords.T, axis=2).reshape((X_OUT*Y_OUT, 2))
        self.diff = None # Difference between inputs and weights (x - w)
        self.bmu_dists = []

        # Initialize weights
        # First layer takes observed cones as input
        self.w1 = np.random.rand(X_OUT, Y_OUT, IN_MAX, IN_VECT)
        self.w1 = self.w1 * [10.0, 15.0] - [5.0, 0.0]
        # Second layer takes BMU of first layer as input
        self.w2 = np.random.rand(X_OUT, Y_OUT, IN_VECT) * [X_OUT, Y_OUT]

        # Set subscribers
        self.cones_sub = self.create_subscription(ConeArray,
            '/cones/positions', self.cones_callback, 10)
        
        # Timers
        self.dt = []
        self.elapsed = []
        self.start = self.get_clock().now().nanoseconds
        self.timer_last = self.get_clock().now().nanoseconds
    
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
        # Calculate time values
        cur_time = self.get_clock().now().nanoseconds
        T = (cur_time - self.start)/1000000000
        DT = (cur_time - self.timer_last)/1000000000
        self.timer_last = cur_time
        self.elapsed.append(T)
        self.dt.append(DT)

        t = self.t
        if (sigma(t) < THRESHOLD):
            print('Training complete')
            if(PLOTTING):
                self.plot_som()
            return
        # Place x y positions of cones into input array x and reshape as 4D
        x = np.array([[cone.x, cone.y] for cone
                        in msg.cones]).reshape((1, 1, len(msg.cones), IN_VECT))
        
        bmu = self.get_bmu(x, self.w1) # First layer BMU coordinates
        
        dist = self.compute_bmu_distances(bmu)
        
        # Compute weights of neighbourhood nodes
        self.w1[:, :, 0:x.shape[2], :] = self.update(self.w1, dist, t)

        # Run 2nd layer with first layer BMU
        bmu = self.get_bmu(np.array(bmu), self.w2)

        dist = self.compute_bmu_distances(bmu)

        self.w2 = self.update(self.w2, dist, t)
        
        self.t += 1 # Increment timestep
        print(bmu)
        print('Quantisation error: ' + str(self.quant_err()))
        print('Radius: ' + str(sigma(t)))
        if(PLOTTING):
            self.plot_som()
        if(DEBUGGING):
            self.debug += 1
            file = 'debug' + str(self.debug) + '.txt'
            f = open(self.path + '/' + file, 'w')
            f.write('Time complexity:\n')
            f.write(str(np.array((self.dt, self.elapsed)).T))
            f.close()

    def get_bmu(self, x, w):
        """
        Determines the X-Y position of the winning node in the first layer

        :param x: The input array (relative x-y pos of visible cones)
        :param w: The weights
        :return: bmu, the coordinates of the best matching unit
        """        
        # Calculate distances between the input and each node

        if (w.ndim == 4):
            w = w[:, :, 0:x.shape[2], :] # Apply mask to w
            results = np.sqrt(np.sum((w - x) * (w - x), axis=(2, 3)))
        elif (w.ndim == 3):
            results = np.sqrt(np.sum((w - x) * (w - x), axis=2))

        self.diff = x - w # Calculate differences
        
        # Get X-Y position of the node with the minimum distance
        bmu = np.unravel_index(np.argmin(results, axis = None), results.shape)
        
        if (w.ndim == 3):
            self.bmu_dists.append(np.linalg.norm(self.diff[bmu]))

        return bmu

    def get_x_y(self):
        """
        Collects x and y weights from the output layer for plotting
        """
        x = self.w2[:, :, 0].flatten()
        y = self.w2[:, :, 1].flatten()

        return x, y

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

    def plot_som(self, bmu = None):
        """
        Plots the SOM by taking x, y weights from the output layer

        :param bmu: If supplied, highlights this unit in red.
        """
        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event', lambda event:
            [exit(0) if event.key == 'escape' else None])
        x, y = self.get_x_y()
        plt.plot(x, y, ".k", label='Units')
        if (bmu):
            plt.plot(self.w2[bmu][0], self.w2[bmu][1], "*r", label='BMU')
        plt.legend()
        plt.title('BioSLAM: SOM')
        plt.xlabel('X Weights')
        plt.ylabel('Y Weights')
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)
    
    def quant_err(self):
        """
        Measures the quantisation error, the average difference between the
        input and the BMU (output layer)

        :return: The quantisation error
        """
        return np.array(self.bmu_dists).mean()

    def update(self, w, dist, t):
        """
        Calculates new weights for the SOM

        :param w: The weights array to be updated
        :param dist: The array of distances for each node from the BMU
        :param t: The given timestep
        :return: The updated weights array
        """

        if (w.ndim == 4):
            w = w[:, :, 0:self.diff.shape[2], :]
            theta_dt = theta(dist, t).reshape((X_OUT, Y_OUT, 1, 1))
        elif (w.ndim == 3):
            theta_dt = theta(dist, t).reshape((X_OUT, Y_OUT, 1))

        w += theta_dt * L(t) * self.diff

        return w


def main(args=None):
    rclpy.init(args=args)

    node = Listener()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv)
