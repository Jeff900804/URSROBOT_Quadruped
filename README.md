# [Quadrupedal Robot Locomotion Control for Obstacle Traversal and Crouching](https://jeff900804.github.io/URSROBOT_Quadruped/)
## Overview
This project presents a locomotion control framework for a quadrupedal robot capable of performing stable velocity tracking, obstacle traversal, and crouching posture transitions.

The controller is designed to enable the robot to adapt its body posture and leg motion according to commanded velocity and local terrain information. In addition to normal walking, the system supports traversing discontinuous terrain or low obstacles and switching to a crouching motion mode when reduced body height is required.

The project is validated in both simulation and real-world experiments using a quadrupedal robot platform.

## Environmemt
### Installing Isaac Sim
You can create the Isaac Lab environment using the following commands.
```bash
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab
```
Ensure the latest pip version is installed. To update pip, run the following command from inside the virtual environment:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
Install Isaac Sim pip packages:
```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```
Install a CUDA-enabled PyTorch build that matches your system architecture:
```bash
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```
### Installing Isaac Lab
#### Cloning Isaac lab
```bash
cd /path/to/URSROBOT_Quadruped
git clone https://github.com/isaac-sim/IsaacLab.git
```
#### Installation
Run the install command that iterates over all the extensions in source directory and installs them using pip
```bash
sudo apt install cmake build-essential
cd IsaacLab/
./isaaclab.sh --install
```
### Installing Unitree RL Lab
Use a python interpreter that has Isaac Lab installed, install the library in editable mode using:
```bash
cd ..
cd unitree_rl_lab
./unitree_rl_lab.sh -i
```
