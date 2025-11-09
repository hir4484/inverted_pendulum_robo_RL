import argparse
import os
import pickle
import shutil

from pend5_env import Pend2Env
from rsl_rl.runners import OnPolicyRunner
import genesis as gs


def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {                     ### PPO algorithm settings
            "clip_param":      0.2,          # Policy update clipping range
            "desired_kl":     0.01,          # KL divergence goal (used to adjust learning rate)
            "entropy_coef":   0.01,          # Entropy bonus coefficient (encourages policy diversity)
            "gamma":          0.99,          # Discount factor
            "lam":            0.95,          # GAE (Generalized Advantage Estimation) parameter
            "learning_rate": 0.001,          # Learning rate
            "max_grad_norm":   1.0,          # Gradient clipping threshold
            "num_learning_epochs": 5,        # Number of training epochs in each update step
            "num_mini_batches":    4,        # Number of mini-batches
            "schedule": "adaptive",          # Learning rate scheduling method (adaptive or fixed)
            "use_clipped_value_loss": True,  # Whether to use clipping in the value function loss calculation
            "value_loss_coef":   1.0,        # Coefficient of the value function loss
        },
        "init_member_classes": {},
        "policy": {                        ### Policy and value network settings
            "activation": "elu",             # Activation function ('elu')
            "actor_hidden_dims":  [512, 256, 128], # Hidden layer dimensions of the policy network (Actor)
            "critic_hidden_dims": [512, 256, 128], # Hidden layer dimensions of the value network (Critic) [256, 128]
            "init_noise_std":  1.0,          # Standard deviation of the initial noise for actions
        },
        "runner": {                        ### Training execution settings
            "algorithm_class_name": "PPO",   # Class name of the algorithm to use ('PPO')
            "checkpoint":        -1,         # Step number to load checkpoints for (-1 means no loading)
            "experiment_name": exp_name,     # Experiment name
            "load_run":          -1,         # Run ID to load (-1 means no loading)
            "log_interval":       1,         # Log output interval (number of steps)
            "max_iterations": max_iterations,# Maximum number of training runs
            "num_steps_per_env": 24,         # Number of data points to collect per step in each environment
            "policy_class_name": "ActorCritic", # Class name that combines the policy and value function ('ActorCritic')
            "record_interval":   -1,         # Video recording interval (-1 means no recording)
            "resume":     False,             # Whether to resume training
            "resume_path": None,             # Path to the checkpoint to resume
            "run_name": "",                  # Run name
            "runner_class_name": "runner_class_name", # Name of the class that manages the training loop
            "save_interval":    100,         # Interval for saving the model
        },
        "runner_class_name": "OnPolicyRunner",     # Class name of the runner
        "seed": 1,                           # Random number seed
    }

    return train_cfg_dict


# Environment (env_cfg), observation (obs_cfg), reward (reward_cfg), and command (command_cfg) configuration
def get_cfgs():
    env_cfg = {                   ### Environment settings
        "num_actions": 1,           # Number of dimensions of the action (number of joints)
        "default_joint_angles": {   # Default joint angle [rad]
            "Joint_gearbox_shaft": 0.0,
        },
        
        "dof_names": [              # joint/link names
            "Joint_gearbox_shaft",  # in this case, only one dof
        ],
        
        # PD coefficient
        "kp": 40.0,                 # Proportional gain
        "kd": 0.11,                 # Derivative gain
        
        # terminate judgment pitch angle
        "termination_if_roll_greater_than":  90.0,  # [degree]
        "termination_if_roll_smaller_than":  0.12,  # [degree]
        
        # base pose
        "base_init_pos":  [0.0, 0.0085, 0.0228],              # Robot initial coordinates
        "base_init_quat": [0.9829, -0.1844, 0.0001, -0.0005], # Robot initial posture => Seated pitch angle: -21.28[degree]
        
        "episode_length_s": 10.0,   # Maximum episode length in seconds
        "resampling_time_s": 4.0,   # Command resampling interval
        
        "action_scale": 0.25,            # Behavioral Scaling Factor
        "simulate_action_latency": True, # Whether to simulate delays in behavior
        "clip_actions": 100.0,           # Behavioral Clipping Range
    }
    obs_cfg = {                   ### Observation settings
        "num_obs": 12,              # Number of observation dimensions
        "obs_scales": {             # A factor for scaling each observation
            "lin_vel": 2.0,
            "ang_vel": 1.0, 
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {                ### Reward Settings
        "tracking_sigma": 0.13,     # 0.15
        "upright_sigma":  0.13,     # 0.5 0.3 0.25(good->17deg) 0.2(13deg) 0.15(19deg)
        "rise_gain":       0.8,     # 1.0 1.2 1.5
        "min_height":     0.15,     # 0.15
        "height_gain":    0.15,     # 0.2
        
        "reward_scales": {              # Reward weighting
            "tracking_lin_vel": 0.40,   # 0.12 0.15 0.30 0.40
            "action_rate":    -0.010,   # -0.005
            # ----------------------------------------
            "upright_pitch":     0.1,   # 1.6(NG) 0.8() 0.6(-14deg) 0.4(15deg) 0.2(16deg) 0.1(12deg) 0.02(OK)
            "rise_motion":       1.2,   # 3.0(good) 2.2(good) 1.2 0.8 0.9 0.3
            "height_gain":       0.07,  # 0.07(a little bit) 0.14(slowly) 0.2 0.6(small movements)
            # ----------------------------------------
        },
    }
    command_cfg = {                   ### Command-related settings
        "num_commands": 3,              # Command Dimension
        "lin_vel_x_range": [ 0.0, 0.0], # Maximum and minimum velocity along the x-axis = 0.0
        "lin_vel_y_range": [-0.1, 0.1], # Maximum and minimum velocity along the y-axis = 0.0
        "ang_vel_range": [-20.0, 20.0], # Maximum and minimum angular velocity
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="param_10") ### put into "output folder name" ###
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=300)        ### put into MAX ITERATION NUM ###
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = Pend2Env(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
"""
# TensorBoard command
# tensorboard --logdir=./
"""