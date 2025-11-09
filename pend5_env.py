import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Pend2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda"):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = True
        self.dt = 0.010
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        
        self.num_sim_substeps = 2

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=self.num_sim_substeps),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos    = (3.5, 0.0, 0.6),
                camera_lookat = (0.0, 0.0, 0.05),
                camera_fov    = 4,
            ),
            vis_options=gs.options.VisOptions(
                n_rendered_envs=1,
                show_world_frame = False,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        ### add plain
        self.scene.add_entity(gs.morphs.Plane())

        ### add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.camera = None
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file='urdf/pendulum_robot_renew/Robot.urdf',
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        if show_viewer:
            # Recording camera settings used "pend5_eval.py"
            self.camera = self.scene.add_camera(
                res=(800, 720),
                pos=(3.5, 0.0, 0.6),
                lookat=(0.0, 0.0, 0.05),
                fov=4,
                GUI=False
            )
        
        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
            
        self.episode_sums["angle_pitch/time_avg"] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        
        self.time_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float) 

        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.time_buf += self.dt 
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length

        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) < self.env_cfg["termination_if_roll_smaller_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0
        
        # Reset the environment
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            # Add the cumulative value of each reward term to episode_sums
            self.episode_sums[name] += rew

        # Accumulation of angle_pitch/time_avg
        # The buffer for reset environments is already reset at reset_idx, so only active environments are accumulated.
        if self.base_euler is not None:
            active_envs = ~self.reset_buf ### Non-reset environment
            self.episode_sums["angle_pitch/time_avg"][active_envs] += self.base_euler[:, 0][active_envs]

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],
                self.projected_gravity,
                self.commands * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
                self.dof_vel * self.obs_scales["dof_vel"],
                self.actions,
            ],
            axis=-1,
        )

        # Buffer Update
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    # Returns the current observation (obs_buf)
    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self): # no use in this time
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers (partial reset: counters are kept for logging)
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        # NOTE: self.episode_length_buf[envs_idx] = 0 moves after logging
        self.reset_buf[envs_idx] = True

        # fill extras
        # Update log information
        self.extras["episode"] = {} # initialize
        
        # Gets the length of the episode just before it was reset (in steps and time)
        finished_episode_length = self.episode_length_buf[envs_idx].float()
        finished_episode_time = self.time_buf[envs_idx]
        
        for key in self.episode_sums.keys():
            #
            # Divide the total accumulated angle by the number of steps to calculate the "AVERAGE PITCH ANGLR / STEP"
            #
            if key == "angle_pitch/time_avg":
                num_steps = finished_episode_length
                valid_envs = num_steps > 0
                
                if torch.any(valid_envs):
                    pitch_sum = self.episode_sums[key][envs_idx][valid_envs]
                    mean_pitch_per_step = pitch_sum / num_steps[valid_envs]
                    # Average across all active environments and log
                    self.extras["episode"]["angle_pitch/time_avg"] = torch.mean(mean_pitch_per_step).item() 
                else:
                    self.extras["episode"]["angle_pitch/time_avg"] = 0.0
                # Reset the accumulated value
                self.episode_sums[key][envs_idx] = 0.0
                
                continue
            
            # Logging reward items
            valid_times = finished_episode_time > 0
            if torch.any(valid_times):
                avg_reward = self.episode_sums[key][envs_idx][valid_times] / finished_episode_time[valid_times]
                self.extras["episode"][key] = torch.mean(avg_reward).item()
            else:
                self.extras["episode"][key] = 0.0

            self.episode_sums[key][envs_idx] = 0.0
            
        # NOTE: Reset counter after logging is complete
        self.episode_length_buf[envs_idx] = 0
        self.time_buf[envs_idx] = 0.0

        # Call "_resample_commands" to update the target velocity
        self._resample_commands(envs_idx)

    # Reset all environments
    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])
    
    
    # --- Model-free inverted search reward ---
    def _reward_upright_pitch(self):
        # The closer to the inverted posture (pitch ≈ 0), the higher the reward.
        pitch_rad = self.base_euler[:, 0]
        pitch_error = torch.abs(pitch_rad)
        return torch.exp(-pitch_error / self.reward_cfg["upright_sigma"])

    def _reward_rise_motion(self):
        # It encourages the standing up movement, and the greater the angular velocity, the greater the reward.
        pitch_rate = torch.abs(self.base_ang_vel[:, 0])
        return torch.clamp(pitch_rate / self.reward_cfg["rise_gain"], max=1.0)

    def _reward_height_gain(self):
        # The higher the center of gravity, the higher the reward.
        height = self.base_pos[:, 2]
        return torch.clamp((height - self.reward_cfg["min_height"]) / self.reward_cfg["height_gain"], max=1.0)
    # ----------------------------------------
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
