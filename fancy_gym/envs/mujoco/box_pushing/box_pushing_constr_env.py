import numpy as np
from fancy_gym.envs.mujoco.box_pushing.box_pushing_env import BoxPushingDense

class BoxPushingConstrDense(BoxPushingDense):
    def __init__(self, **kwargs):
        super(BoxPushingConstrDense, self).__init__(**kwargs)
    def _get_reward(self, episode_end, box_pos, box_quat, target_pos, target_quat,
                    rod_tip_pos, rod_quat, qpos, qvel, action):
        #joint_penalty = self._joint_limit_violate_penalty(qpos,
        #                                                  qvel,
        #                                                  enable_pos_limit=True,
        #                                                  enable_vel_limit=True)
        #tcp_box_dist_reward = -2 * np.clip(np.linalg.norm(box_pos - rod_tip_pos), 0.05, 100)
        #box_goal_pos_dist_reward = -3.5 * np.linalg.norm(box_pos - target_pos)
        #box_goal_rot_dist_reward = -rotation_distance(box_quat, target_quat) / np.pi
        #energy_cost = -0.0005 * np.sum(np.square(action))

        #reward = joint_penalty + tcp_box_dist_reward + \
        #    box_goal_pos_dist_reward + box_goal_rot_dist_reward + energy_cost

        #rod_inclined_angle = rotation_distance(rod_quat, self._desired_rod_quat)
        #if rod_inclined_angle > np.pi / 4:
        #    reward -= rod_inclined_angle / (np.pi)

        reward = 0.
        print("OUR ENV")
        return reward