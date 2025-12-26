from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_conjugate
from isaaclab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg

def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase

class MotionObservation(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset: Articulation = env.scene[asset_cfg.name]

        feet_body_names = cfg.params["feet_body_names"]
        elbow_body_names = cfg.params["elbow_body_names"]
        mujoco_joint_names = cfg.params["mujoco_joint_names"]
        mujoco_leftleg_names = cfg.params["mujoco_leftleg_names"]
        mujoco_leftarm_names = cfg.params["mujoco_leftarm_names"]
        mujoco_rightleg_names = cfg.params["mujoco_rightleg_names"]
        mujoco_rightarm_names = cfg.params["mujoco_rightarm_names"]
        mujoco_ankle_names = cfg.params["mujoco_ankle_names"]


        self.feet_body_ids, _ = asset.find_bodies(
            name_keys=feet_body_names, preserve_order=True
        )
        self.elbow_body_ids, _ = asset.find_bodies(
            name_keys=elbow_body_names, preserve_order=True
        )

        ##
        self.mujoco_joint_ids, _ = asset.find_joints(
            name_keys=mujoco_joint_names,
            preserve_order=True,
        )

        self.left_leg_ids, _ = asset.find_joints(
            name_keys=mujoco_leftleg_names,
            preserve_order=True,
        )
        self.right_leg_ids, _ = asset.find_joints(
            name_keys=mujoco_rightleg_names,
            preserve_order=True,
        )
        self.left_arm_ids, _ = asset.find_joints(
            name_keys=mujoco_leftarm_names,
            preserve_order=True,
        )
        self.right_arm_ids, _ = asset.find_joints(
            name_keys=mujoco_rightarm_names,
            preserve_order=True,
        )
        self.ankle_joint_ids, _ = asset.find_joints(
            name_keys=mujoco_ankle_names,
            preserve_order=True,
        )

        self.left_arm_local_vec = torch.tensor([0.0, 0.0, -0.3], device=self.device).repeat((self.num_envs, 1))
        self.right_arm_local_vec = torch.tensor([0.0, 0.0, -0.3], device=self.device).repeat((self.num_envs, 1))


    def __call__(self,
            env: ManagerBasedRLEnv,
            asset_cfg: SceneEntityCfg,

            feet_body_names: str,
            elbow_body_names: str,
            mujoco_joint_names: str,
            mujoco_leftleg_names: str,
            mujoco_leftarm_names: str,
            mujoco_rightleg_names: str,
            mujoco_rightarm_names: str,
            mujoco_ankle_names: str
        ) -> torch.Tensor:

            asset: Articulation = env.scene[asset_cfg.name]

            left_hand_pos = (
                asset.data.body_state_w[:, self.elbow_body_ids[0], :3]
                - asset.data.root_state_w[:, 0:3]
                + quat_apply(asset.data.body_state_w[:, self.elbow_body_ids[0], 3:7], self.left_arm_local_vec)
            )
            right_hand_pos = (
                asset.data.body_state_w[:, self.elbow_body_ids[1], :3]
                - asset.data.root_state_w[:, 0:3]
                + quat_apply(asset.data.body_state_w[:, self.elbow_body_ids[1], 3:7], self.right_arm_local_vec)
            )
            left_hand_pos = quat_apply(quat_conjugate(asset.data.root_state_w[:, 3:7]), left_hand_pos)
            right_hand_pos = quat_apply(quat_conjugate(asset.data.root_state_w[:, 3:7]), right_hand_pos)
            left_foot_pos = (
                asset.data.body_state_w[:, self.feet_body_ids[0], :3] - asset.data.root_state_w[:, 0:3]
            )
            right_foot_pos = (
                asset.data.body_state_w[:, self.feet_body_ids[1], :3] - asset.data.root_state_w[:, 0:3]
            )
            left_foot_pos = quat_apply(quat_conjugate(asset.data.root_state_w[:, 3:7]), left_foot_pos)
            right_foot_pos = quat_apply(quat_conjugate(asset.data.root_state_w[:, 3:7]), right_foot_pos)


            left_leg_dof_pos = asset.data.joint_pos[:, self.left_leg_ids]
            right_leg_dof_pos = asset.data.joint_pos[:, self.right_leg_ids]
            left_leg_dof_vel = asset.data.joint_vel[:, self.left_leg_ids]
            right_leg_dof_vel = asset.data.joint_vel[:, self.right_leg_ids]
            left_arm_dof_pos = asset.data.joint_pos[:, self.left_arm_ids]
            right_arm_dof_pos = asset.data.joint_pos[:, self.right_arm_ids]
            left_arm_dof_vel = asset.data.joint_vel[:, self.left_arm_ids]
            right_arm_dof_vel = asset.data.joint_vel[:, self.right_arm_ids]

            return torch.cat(
                (
                    right_arm_dof_pos,
                    left_arm_dof_pos,
                    right_leg_dof_pos,
                    left_leg_dof_pos,
                    right_arm_dof_vel,
                    left_arm_dof_vel,
                    right_leg_dof_vel,
                    left_leg_dof_vel,
                    left_hand_pos,
                    right_hand_pos,
                    left_foot_pos,
                    right_foot_pos,
                ),
                dim=-1,
            )
