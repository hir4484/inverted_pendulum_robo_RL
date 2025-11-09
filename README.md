## Reinforcement learning file for a two-wheeled inverted pendulum running in the Genesis environment<br>
<br>
This file is a repository of deep reinforcement learning training files explained in the YouTube video "Deep Reinforcement Learning and Model-Free Optimal Control in Genesis Simulator." It is intended for use with the "Genesis" physics platform.<br>
<br>
For the YouTube explanation, please refer to the following URL:<br>
https://www.youtube.com/watch?v=me3NUO_06fw<br>
<img src="./youtube_robo6th.png" width="500"><br>
<br>
About the file contents:<br>
- Inverted control of a two-wheeled inverted pendulum model is performed using deep reinforcement learning in the Genesis simulation environment.<br>
- A reinforcement learning reward function that achieves "model-free optimal control" is created and demonstrated.<br>
- A reward structure that explores the inverted posture based on interaction with the environment is implemented.<br>
<br>
The following three files are used for reinforcement learning in Genesis.<br>
1. env.py (Environment Definition File)<br>
: Defines the environment in which the reinforcement learning agent interacts.<br>
2. train.py (Training Executable File)<br>
: This is the main script that trains the agent using the reinforcement learning algorithm.<br>
3. eval.py (Evaluation/Test File)<br>
: This is a script for evaluating the performance of a trained model.<br>
<br>
Please download the robot model file (URDF) from the repository below.<br>
https://github.com/hir4484/inverted_pendulum_sim_in_Genesis<br>
You'll find this model file in the pendulum_robot_renew folder.<br>
<img src="./robot.png" width="400"><br>
<br>
To run training, use the following command:<br>
~$ python3 pend5_train.py<br>
<br>
When running evaluation after training, you can record the screen by adding an option to the eval file.<br>
To record:<br>
~$ python3 pend5_eval.py -R<br>
~$ python3 pend5_eval.py --record<br>
To not record:<br>
~$ python3 pend5_eval.py<br>
<br>
To start TensorBoard:<br>
~$ tensorboard --logdir=./<br>
<img src="./tensorboard_param10.png" width="800"><br>
<br>
Copyright (c) 2025/Nov/09, hir (hir4484@gmail.com). Available under the MIT License.
