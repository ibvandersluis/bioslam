# BioSLAM

This repository is one of two parts of my dissertation for my master's at Oxford Brookes University. See also my [repository for making FastSLAM](https://github.com/ibvandersluis/fastslam), the other part of my dissertation.

## Description

This is an implementation of biologically-inspired simultaneous localisation and mapping (SLAM) for a self-driving car.

BioSLAM works through the use of a Kohonen network, a type of neural network used for unsupervised competitive learning that results in a self-organising map (SOM). As a winner-take-all network, only one node (called the best matching unit, or BMU) in the output layer wins at each timestep. The BMU has its weights adjusted the most, and the 'neighbourhood' of nearby nodes have weights adjusted proportionate to their distance from the BMU, where nearby nodes are adjusted considerably and far off nodes are adjusted just barely or not at all.

The design for this implementation was mostly inspired by these sources:
- [AI-Junkie's C++ tutorial for a self-organising feature map](http://www.ai-junkie.com/ann/som/som1.html)
- ['Having Fun with Self-Organising Maps'](https://tcosmo.github.io/2017/07/27/fun-with-som.html) by Cosmo's Blog
- Chapter 9 of [_Introduction to the Theory of Neural Computation_](https://www.amazon.co.uk/Introduction-Computation-Institute-Sciences-Complexity/dp/0201515601) by Hertz, Krogh, and Palmer, 1991

## Requirements

- Assumes an existing ROS 2 installation (Dashing or newer). If you are using a distro other than Dashing, replace 'dashing' in the following terminal commands with your distro name.
    - [Install ROS 2 Dashing](https://index.ros.org/doc/ros2/Installation/Dashing/)
- This package also uses the following Python3 libraries:
    - NumPy
    - SciPy
    - Matplotlib

## Installation

0. Open a terminal and source your ROS installation

```bash
source /opt/ros/dashing/setup.bash
```

1. Create a new directory for your ROS 2 workspace (if you won't be using an existing one). For instance, if you named your workspace `ws` and placed it in your home directory:

```bash
mkdir -p ~/ws/src
```

2. Clone this repository into the `src` directory of your ROS 2 workspace

```bash
cd ~/ws/src
git clone https://github.com/ibvandersluis/bioslam.git
```

3. Also clone dependent packages

```bash
git clone https://gitlab.com/obr-a/integration/ads_dv_msgs.git
git clone https://gitlab.com/obr-a/integration/obr_msgs.git
git clone https://gitlab.com/obr-a/integration/helpers.git
```

4. Resolve package dependencies

```bash
cd ~/ws
rosdep install -i --from-path src --rosdistro dashing -y
```

5. Build (this will take a couple minutes when building packages for the first time)

```bash
colcon build --symlink-install
```

## Usage

After you have run `colcon build`:
1. Open 2 new tabs in your terminal and source the workspace in each

```bash
cd ~/ws
. install/setup.bash
```

2. In one of the new tabs play the rosbag. Some rosbags are included in the `fastslam/bags` directory

```bash
ros2 bag play <path_to_rosbag>
```

3. In the other new tab run the FastSLAM node

```bash
ros2 run bioslam main
```