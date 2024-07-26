import numpy as np
import mujoco
from gymnasium.spaces import Box
from fancy_gym.envs.mujoco.box_pushing.box_pushing_env import BoxPushingDense, MAX_EPISODE_STEPS_BOX_PUSHING 
from fancy_gym.envs.mujoco.box_pushing.box_pushing_utils import rot_to_quat, get_quaternion_error, rotation_distance
from fancy_gym.envs.mujoco.box_pushing.box_pushing_utils import rot_to_quat, get_quaternion_error, rotation_distance
from fancy_gym.envs.mujoco.box_pushing.box_pushing_utils import q_max, q_min, q_dot_max, q_torque_max
from fancy_gym.envs.mujoco.box_pushing.box_pushing_utils import desired_rod_quat
from spline_rl.utils.constants import BOX_PUSHING_ROD_MINIMUM_HEIGHT
from spline_rl.utils.constraints import BoxPushingConstraints

import matplotlib.pyplot as plt
#from storm_kit.differentiable_robot_model.differentiable_robot_model import DifferentiableRobotModel

class BoxPushingConstrDense(BoxPushingDense):
    def __init__(self, horizon=MAX_EPISODE_STEPS_BOX_PUSHING, **kwargs):
        super(BoxPushingConstrDense, self).__init__(**kwargs)
        self.horizon = horizon
        self.env_info = {
            "dt": self.dt,
            "robot": {
                "joint_pos_limit": np.array([q_min, q_max]),
                "joint_vel_limit": np.array([-q_dot_max, q_dot_max]),
                "joint_acc_limit": np.array([-10. * np.ones_like(q_dot_max), 10. * np.ones_like(q_dot_max)]),
                "joint_torque_limit": np.array([-q_torque_max, q_torque_max]), 
                "robot_model": self.model,
                "robot_data": self.data,
                "n_joints": 7,
            },
        }
        self.constraints = BoxPushingConstraints(
            self.env_info['robot']['joint_pos_limit'][0],
            self.env_info['robot']['joint_pos_limit'][1],
            self.env_info['robot']['joint_vel_limit'][1],
            self.env_info['robot']['joint_acc_limit'][1],
        )
        #self.robot = DifferentiableRobotModel(urdf_path="../urdf/franka.urdf", name="franka")
        box_pose_min = np.array([0.3, -0.45, -0.01, 0., 0., 0., -1.])
        box_pose_max = np.array([0.6, 0.45, -0.01, 1., 0., 0., 1.])
        self.observation_space = Box(
            low=np.concatenate([q_min, -q_dot_max, box_pose_min, box_pose_min]),
            high=np.concatenate([q_max, q_dot_max, box_pose_max, box_pose_max]),
            shape=(28,), dtype=np.float64
        )
        #self.action_space = Box(-q_torque_max, q_torque_max, shape=(7,), dtype=np.float64)
        self.action_space = Box(np.stack([q_min, -q_dot_max, self.env_info['robot']['joint_acc_limit'][0]]),
                                np.stack([q_max, q_dot_max, self.env_info['robot']['joint_acc_limit'][1]]),
                                shape=(3, 7), dtype=np.float64)

    def _compute_action(self, desired_trajectory):
        cur_pos, cur_vel = self.data.qpos[:7].copy(), self.data.qvel[:7].copy()
        desired_pos = desired_trajectory[0]
        desired_vel = desired_trajectory[1]
        desired_acc = desired_trajectory[2]
        return self._controller(desired_pos, desired_vel, desired_acc, cur_pos, cur_vel)

    def _get_reward(self, episode_end, box_pos, box_quat, target_pos, target_quat,
                    rod_tip_pos, rod_quat, qpos, qvel, action):
        tcp_box_dist_reward = -2 * np.clip(np.linalg.norm(box_pos - rod_tip_pos), 0.05, 100)
        box_goal_pos_dist_reward = -3.5 * np.linalg.norm(box_pos - target_pos)
        box_goal_rot_dist_reward = -rotation_distance(box_quat, target_quat) / np.pi
        energy_cost = -0.0005 * np.sum(np.square(action))

        #reward = tcp_box_dist_reward + box_goal_pos_dist_reward + box_goal_rot_dist_reward + energy_cost
        reward = box_goal_pos_dist_reward + box_goal_rot_dist_reward

        rod_inclined_angle = rotation_distance(rod_quat, self._desired_rod_quat)
        if rod_inclined_angle > np.pi / 4:
            reward -= rod_inclined_angle / (np.pi)
        return reward

    
    def _get_constraints(self, rod_tip_pos, qpos, qvel):
        rod_tip_pos_constraint = np.maximum(BOX_PUSHING_ROD_MINIMUM_HEIGHT - rod_tip_pos[2], 0)
        qpos_lb = np.maximum(q_min - qpos, 0)
        qpos_ub = np.maximum(qpos - q_max, 0)
        qpos_constraint = np.maximum(qpos_ub, qpos_lb)
        qvel_lb = np.maximum(-q_dot_max - qvel, 0)
        qvel_ub = np.maximum(qvel - q_dot_max, 0)
        qvel_constraint = np.maximum(qvel_ub, qvel_lb)
        return rod_tip_pos_constraint, qpos_constraint, qvel_constraint
        
    
    def reset(self, *args, **kwargs):
        r = super(BoxPushingConstrDense, self).reset(*args, **kwargs)
        #self.data.qvel[0] = 0.08
        self.qs = []
        self.qs_desired = []
        return r

    def step(self, desired_trajectory):

        unstable_simulation = False
        self.velocity_profile.append(self.data.qvel[:7].copy())

        resultant_actions = []
        for _ in range(self.frame_skip):
            try:
                self.qs.append(self.data.qpos[:7].copy())
                self.qs_desired.append(desired_trajectory[0])
                control_action = self._compute_action(desired_trajectory)
                resultant_action = np.clip(control_action, -q_torque_max, q_torque_max)
                self.do_simulation(resultant_action, 1)
                resultant_actions.append(resultant_action)
            except Exception as e:
                print(e)
                unstable_simulation = True

        self._steps += 1
        self._episode_energy += np.sum(np.square(resultant_actions))

        episode_end = True if self._steps >= self.horizon else False

        box_pos = self.data.body("box_0").xpos.copy()
        box_quat = self.data.body("box_0").xquat.copy()
        target_pos = self.data.body("replan_target_pos").xpos.copy()
        target_quat = self.data.body("replan_target_pos").xquat.copy()
        rod_tip_pos = self.data.site("rod_tip").xpos.copy()
        rod_quat = self.data.body("push_rod").xquat.copy()
        qpos = self.data.qpos[:7].copy()
        qvel = self.data.qvel[:7].copy()

        if not unstable_simulation:
            reward = self._get_reward(episode_end, box_pos, box_quat, target_pos, target_quat,
                                      rod_tip_pos, rod_quat, qpos, qvel, resultant_actions)
        else:
            reward = -50

        rod_tip_pos_constraint, qpos_constraint, qvel_constraint = self._get_constraints(rod_tip_pos, qpos, qvel)

        obs = self._get_obs()
        box_goal_pos_dist = 0. if not episode_end else np.linalg.norm(box_pos - target_pos)
        box_goal_quat_dist = 0. if not episode_end else rotation_distance(box_quat, target_quat)
        mean_squared_jerk, maximum_jerk, dimensionless_jerk = (0.0,0.0,0.0) if not episode_end else self.calculate_smoothness_metrics(np.array(self.velocity_profile), self.dt)
        infos = {
            'episode_end': episode_end,
            'box_goal_pos_dist': box_goal_pos_dist,
            'box_goal_rot_dist': box_goal_quat_dist,
            'episode_energy': 0. if not episode_end else self._episode_energy,
            'mean_squared_jerk': mean_squared_jerk,
            'maximum_jerk': maximum_jerk,
            'dimensionless_jerk': dimensionless_jerk,
            'success': 1. if episode_end and box_goal_pos_dist < 0.05 and box_goal_quat_dist < 0.5 else 0.,
            'num_steps': self._steps,
            'rod_tip_pos_constraint': rod_tip_pos_constraint,
            'qpos_constraint': qpos_constraint,
            'qvel_constraint': qvel_constraint,
        }

        #if self._steps % 50 == 0:
        #    qs_ = np.array(self.qs)
        #    qs_desired_ = np.array(self.qs_desired)
        #    for i in range(6):
        #        plt.subplot(321 + i)
        #        plt.plot(qs_[:, i], label=f'q_{i}')
        #        plt.plot(qs_desired_[:, i], label=f'q_d_{i}')
        #        plt.legend()
        #    plt.show()
        

        if self.render_active and self.render_mode=='human':
            self.render()

        return obs, reward, episode_end, None, infos

    # HOTFIX to enable actions in form of the desired pos, vel, acc and applying resultant torques
    # -> controller is a part of the environment to account for the frameskip
    def do_simulation(self, ctrl, n_frames):
        """
        Step the simulation n number of frames and applying a control action.
        """
        # Check control input is contained in the action space
        if np.array(ctrl).shape != self.data.qpos[:7].shape:
            raise ValueError(
                f"Action dimension mismatch. Expected {self.data.qpos[:7].shape}, found {np.array(ctrl).shape}"
            )
        self._step_mujoco_simulation(ctrl, n_frames)

class BoxPushingConstrDensePDFF(BoxPushingConstrDense):
    def __init__(self, full_mass_matrix=True, **kwargs):
        super(BoxPushingConstrDensePDFF, self).__init__(**kwargs)
        self.full_mass_matrix = full_mass_matrix
        self.n_dof = 7
        self.prev_controller_cmd_pos = np.zeros(self.n_dof)
        #self.action_space = Box(np.concatenate([q_min, -q_dot_max, -q_torque_max]), np.concatenate([q_max, q_dot_max, q_torque_max]))
        n = 1.
        self.p_gain = np.array([1500., 1500., 1200., 1200., 1000., 1000., 500.])
        self.p_gain *= n
        #self.d_gain = np.array([800, 80, 60, 30, 10, 1, 0.5])
        self.d_gain = np.array([80., 80., 60., 30., 10., 10., 5.])
        self.d_gain *= n 
        #self.p_gain = np.array([150., 150., 120., 120., 100., 100., 50.])
        #self.d_gain = np.array([6, 8, 6, 3, 1, 0.1, 0.05])
        #self.p_gain = np.array([10., 10., 10., 10., 10., 10., 10.])
        #self.d_gain = np.array([1, 1, 1, 1, 1, 1, 1])
        #self.p_gain = np.array([1., 1., 1., 1., 1., 1., 1.])
        #self.d_gain = np.zeros(7)

    def _controller(self, desired_pos, desired_vel, desired_acc, current_pos, current_vel):
        #clipped_pos, clipped_vel = self._enforce_safety_limits(desired_pos, desired_vel)
        clipped_pos, clipped_vel = desired_pos, desired_vel

        error = (clipped_pos - current_pos)
        error_dot = (clipped_vel - current_vel)

        torque = self.p_gain * error + self.d_gain * error_dot
        #pd = self.p_gain * error #+ self.d_gain * error_dot
        p = self.p_gain * error
        d = self.d_gain * error_dot
        pd = p + d

        #if self._steps % 30 == 0:
        #    a = 0
        #torque = np.where(np.isfinite(torque), torque, np.zeros_like(torque))

        # Acceleration FeedForward
        #tau_ff = np.zeros(self.n_dof)
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_fullM(self.model, M, self.data.qM)
        #mujoco.mj_mulM(self.robot_model, self.robot_data, tau_ff, desired_acc)

        #torque = np.einsum('ij,j->i', M[:self.n_dof, :self.n_dof], pd)

        # full mass matrix
        if self.full_mass_matrix:
            torque = np.einsum('ij,j->i', M[:self.n_dof, :self.n_dof], desired_acc + pd)
        # diagonal mass matrix
        else:
            torque = np.diag(M[:self.n_dof, :self.n_dof]) * (desired_acc + pd)

        # Gravity Compensation and Coriolis and Centrifugal force
        torque += self.data.qfrc_bias[:self.n_dof]
        #torque = self.data.qfrc_bias[:self.n_dof]
        return torque
    
    #def _enforce_safety_limits(self, desired_pos, desired_vel):
    #    # ROS safe controller
    #    pos = self.prev_controller_cmd_pos
    #    k = 20

    #    joint_pos_lim = np.stack([q_min, q_max], axis=0)
    #    joint_vel_lim = np.stack([-q_dot_max, q_dot_max], axis=0)

    #    min_vel = np.minimum(np.maximum(-k * (pos - joint_pos_lim[0]), joint_vel_lim[0]), joint_vel_lim[1])

    #    max_vel = np.minimum(np.maximum(-k * (pos - joint_pos_lim[1]), joint_vel_lim[0]), joint_vel_lim[1])

    #    clipped_vel = np.minimum(np.maximum(desired_vel, min_vel), max_vel)

    #    min_pos = pos + min_vel * self.dt
    #    max_pos = pos + max_vel * self.dt

    #    clipped_pos = np.minimum(np.maximum(desired_pos, min_pos), max_pos)
    #    self.prev_controller_cmd_pos = clipped_pos.copy()

    #    return clipped_pos, clipped_vel