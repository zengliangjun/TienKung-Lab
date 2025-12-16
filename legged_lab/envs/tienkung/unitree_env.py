from . import tienkung_env

import numpy as np
import torch
from isaaclab.utils.buffers import DelayBuffer
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_conjugate
from scipy.spatial.transform import Rotation

class UnitreeEnv(tienkung_env.TienKungEnv):
    def __init__(
        self,
        cfg,
        headless,
    ):
        super(UnitreeEnv, self).__init__(cfg, headless)



    def init_buffers(self):
        self.extras = {}

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        self.action_scale = self.cfg.robot.action_scale
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)

        self.feet_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
        )
        self.elbow_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_elbow_link", "right_elbow_link"], preserve_order=True
        )

        ##
        self.mujoco_joint_ids, _ = self.robot.find_joints(
            name_keys=self.cfg.mujoco_joint_names,
            preserve_order=True,
        )

        self.left_leg_ids, _ = self.robot.find_joints(
            name_keys=self.cfg.mujoco_leftleg_names,
            preserve_order=True,
        )
        self.right_leg_ids, _ = self.robot.find_joints(
            name_keys=self.cfg.mujoco_rightleg_names,
            preserve_order=True,
        )
        self.left_arm_ids, _ = self.robot.find_joints(
            name_keys=self.cfg.mujoco_leftarm_names,
            preserve_order=True,
        )
        self.right_arm_ids, _ = self.robot.find_joints(
            name_keys=self.cfg.mujoco_rightarm_names,
            preserve_order=True,
        )
        self.ankle_joint_ids, _ = self.robot.find_joints(
            name_keys=["left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_roll_joint"],
            preserve_order=True,
        )
        ########################

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.left_arm_local_vec = torch.tensor([0.0, 0.0, -0.3], device=self.device).repeat((self.num_envs, 1))
        self.right_arm_local_vec = torch.tensor([0.0, 0.0, -0.3], device=self.device).repeat((self.num_envs, 1))

        # Init gait parameter
        self.gait_phase = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.gait_cycle = torch.full(
            (self.num_envs,), self.cfg.gait.gait_cycle, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.phase_ratio = torch.tensor(
            [self.cfg.gait.gait_air_ratio_l, self.cfg.gait.gait_air_ratio_r], dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)
        self.phase_offset = torch.tensor(
            [self.cfg.gait.gait_phase_offset_l, self.cfg.gait.gait_phase_offset_r],
            dtype=torch.float,
            device=self.device,
        ).repeat(self.num_envs, 1)
        self.action = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.init_obs_buffer()



    def visualize_motion(self, time):
        """
        Update the robot simulation state based on the AMP motion capture data at a given time.

        This function sets the joint positions and velocities, root position and orientation,
        and linear/angular velocities according to the AMP motion frame at the specified time,
        then steps the simulation and updates the scene.

        Args:
            time (float): The time (in seconds) at which to fetch the AMP motion frame.

        Returns:
            None
        """
        visual_motion_frame = self.amp_loader_display.get_full_frame_at_time(0, time)
        device = self.device

        dof_pos = torch.zeros((self.num_envs, self.robot.num_joints), device=device)
        dof_vel = torch.zeros((self.num_envs, self.robot.num_joints), device=device)
        # dof
        dof_start_id = 6
        dof_pos[:, self.left_leg_ids] = visual_motion_frame[dof_start_id: dof_start_id + len(self.left_leg_ids)]
        dof_pos[:, self.right_leg_ids] = visual_motion_frame[dof_start_id + len(self.left_leg_ids): dof_start_id + len(self.left_leg_ids) * 2]

        left_dof_arm_start_id = dof_start_id + len(self.left_leg_ids) * 2 + len(self.cfg.mujoco_waist_names)
        dof_pos[:, self.left_arm_ids] = visual_motion_frame[left_dof_arm_start_id: left_dof_arm_start_id + len(self.left_arm_ids)]

        right_dof_arm_start_id = left_dof_arm_start_id + len(self.left_arm_ids) + len(self.cfg.mujoco_lefthand_names)
        dof_pos[:, self.right_arm_ids] = visual_motion_frame[right_dof_arm_start_id: right_dof_arm_start_id + len(self.right_arm_ids)]

        # vel
        vel_start_id = right_dof_arm_start_id + len(self.left_arm_ids) + 6

        dof_vel[:, self.left_leg_ids] = visual_motion_frame[vel_start_id : vel_start_id + len(self.left_leg_ids)]
        dof_vel[:, self.right_leg_ids] = visual_motion_frame[vel_start_id + len(self.left_leg_ids): vel_start_id + len(self.left_leg_ids) * 2]

        left_vel_arm_start_id = vel_start_id + len(self.left_leg_ids) * 2 + len(self.cfg.mujoco_waist_names)
        dof_vel[:, self.left_arm_ids] = visual_motion_frame[left_vel_arm_start_id: left_vel_arm_start_id + len(self.left_arm_ids)]

        right_vel_arm_start_id = left_vel_arm_start_id + len(self.left_arm_ids) + len(self.cfg.mujoco_lefthand_names)
        dof_vel[:, self.right_arm_ids] = visual_motion_frame[right_vel_arm_start_id: right_vel_arm_start_id + len(self.right_arm_ids)]


        self.robot.write_joint_position_to_sim(dof_pos)
        self.robot.write_joint_velocity_to_sim(dof_vel)

        env_ids = torch.arange(self.num_envs, device=device)

        root_pos = visual_motion_frame[:3].clone()
        root_pos[2] += 0.3

        euler = visual_motion_frame[3:6].cpu().numpy()
        quat_xyzw = Rotation.from_euler("XYZ", euler, degrees=False).as_quat()  # [x, y, z, w]
        quat_wxyz = torch.tensor(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=torch.float32, device=device
        )

        lin_vel = visual_motion_frame[6 + self.robot.num_joints: 9 + self.robot.num_joints].clone()
        ang_vel = torch.zeros_like(lin_vel)

        # root state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        root_state = torch.zeros((self.num_envs, 13), device=device)
        root_state[:, 0:3] = torch.tile(root_pos.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 3:7] = torch.tile(quat_wxyz.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 7:10] = torch.tile(lin_vel.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 10:13] = torch.tile(ang_vel.unsqueeze(0), (self.num_envs, 1))

        self.robot.write_root_state_to_sim(root_state, env_ids)
        self.sim.render()
        self.sim.step()
        self.scene.update(dt=self.step_dt)

        left_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[0], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_apply(self.robot.data.body_state_w[:, self.elbow_body_ids[0], 3:7], self.left_arm_local_vec)
        )
        right_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[1], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_apply(self.robot.data.body_state_w[:, self.elbow_body_ids[1], 3:7], self.right_arm_local_vec)
        )
        left_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_hand_pos)
        right_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_hand_pos)
        left_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[0], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        right_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[1], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        left_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_foot_pos)
        right_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_foot_pos)

        self.left_leg_dof_pos =  dof_pos[:, self.left_leg_ids]
        self.right_leg_dof_pos = dof_pos[:, self.right_leg_ids]
        self.left_leg_dof_vel =  dof_vel[:, self.left_leg_ids]
        self.right_leg_dof_vel = dof_vel[:, self.right_leg_ids]
        self.left_arm_dof_pos =  dof_pos[:, self.left_arm_ids]
        self.right_arm_dof_pos = dof_pos[:, self.right_arm_ids]
        self.left_arm_dof_vel =  dof_vel[:, self.left_arm_ids]
        self.right_arm_dof_vel = dof_vel[:, self.right_arm_ids]
        return torch.cat(
            (
                self.right_arm_dof_pos,
                self.left_arm_dof_pos,
                self.right_leg_dof_pos,
                self.left_leg_dof_pos,
                self.right_arm_dof_vel,
                self.left_arm_dof_vel,
                self.right_leg_dof_vel,
                self.left_leg_dof_vel,
                left_hand_pos,
                right_hand_pos,
                left_foot_pos,
                right_foot_pos
            ),
            dim=-1,
        )
