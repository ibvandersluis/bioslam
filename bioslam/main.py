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
from brookes_msgs.msg import Cone, CarPos, ConeArray, IMU, Label
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Point, Twist, Vector3
from gazebo_msgs.msg import LinkStates
from std_msgs.msg import String


class Listener(BaseListener):

    def __init__(self):
        super().__init__('bioslam')

        # Set publishers
        self.map_pub = self.create_publisher(ConeArray, '/mapping/map', 10)
        self.pose_pub = self.create_publisher(CarPos, '/mapping/position', 10)
        self.cmd_pub = self.create_publisher(Twist, '/gazebo/cmd_vel', 10)

        # Set subscribers
        self.cones_sub = self.create_subscription(ConeArray, '/cones/positions', self.cones_callback, 10)
        self.gnss_sub = self.create_subscription(NavSatFix, '/peak_gps/gps', self.gnss_callback, 10)
        self.imu_sub = self.create_subscription(IMU, '/peak_gps/imu', self.imu_callback, 10)
        self.control_sub = self.create_subscription(Twist, '/gazebo/cmd_vel', self.control_callback, 10)

    def cones_callback(self, msg: ConeArray):
        # Place x y positions of cones into self.capture
        self.capture = np.array([[cone.x, cone.y] for cone in msg.cones])
        print(self.capture)

    def control_callback(self, msg: Twist):
        str(msg) # For some reason this is needed to access msg.linear.x
        self.v = msg.linear.x
        self.yaw = msg.angular.z
        self.u = np.array([self.v, self.yaw]).reshape(2, 1)

        self.get_logger().info(f'Command confirmed: {msg.linear.x} m/s turning at {msg.angular.z} rad/s')

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
