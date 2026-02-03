
from . import walk_cfg
from isaaclab.utils import configclass

from legged_lab.assets.unitree import unitree_g123dof
from legged_lab.envs.tienkung import g123dof_info

@configclass
class G123WalkFlatEnvCfg(walk_cfg.TienKungWalkFlatEnvCfg):

    mujoco_joint_names = g123dof_info.mujoco_joint_names

    mujoco_leftleg_names = g123dof_info.mujoco_leftleg_names

    mujoco_rightleg_names = g123dof_info.mujoco_rightleg_names

    mujoco_waist_names = g123dof_info.mujoco_waist_names

    mujoco_leftarm_names = g123dof_info.mujoco_leftarm_names

    mujoco_lefthand_names = g123dof_info.mujoco_lefthand_names

    mujoco_rightarm_names = g123dof_info.mujoco_rightarm_names

    mujoco_righthand_names = g123dof_info.mujoco_righthand_names

    mujoco_ankle_names = g123dof_info.mujoco_ankle_names

    def __post_init__(self):
        super(G123WalkFlatEnvCfg, self).__post_init__()
        #self.amp_motion_files_display = ["legged_lab/envs/tienkung/datasets/motion_visualization/g1_23dof_s1_walking1_stageii.txt"]

        self.scene.robot = unitree_g123dof.UNITREE_G1_23DOF_CFG
        self.reward.undesired_contacts.params["sensor_cfg"].body_names = [".*_knee_link", ".*_shoulder_roll_link", ".*_elbow_link", "pelvis"]
        self.reward.feet_slide.params["sensor_cfg"].body_names = [".*ankle_roll_link"]
        self.reward.feet_slide.params["asset_cfg"].body_names = [".*ankle_roll_link"]
        self.reward.feet_force.params["sensor_cfg"].body_names = [".*ankle_roll_link"]
        self.reward.feet_too_near.params["asset_cfg"].body_names = [".*ankle_roll_link"]
        self.reward.feet_stumble.params["sensor_cfg"].body_names = [".*ankle_roll_link"]


        self.reward.joint_deviation_hip.params["asset_cfg"].joint_names = [ ".*_hip_yaw_joint",
                    ".*_hip_roll_joint",
                    ".*_shoulder_pitch_joint",
                    ".*_elbow_joint"]
        self.reward.joint_deviation_arms.params["asset_cfg"].joint_names = [ ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint"]
        self.reward.joint_deviation_legs.params["asset_cfg"].joint_names = [ ".*_hip_pitch_joint",
                    ".*_knee_joint",
                    ".*_ankle_pitch_joint",
                    ".*_ankle_roll_joint"]

        self.reward.hip_roll_action.weight = -0.15
        self.reward.hip_yaw_action.weight = -0.15

        self.robot.terminate_contacts_body_names=[".*_knee_link", ".*_shoulder_roll_link", ".*_elbow_link", "pelvis"]
        self.robot.feet_body_names=[".*ankle_roll_link"]


@configclass
class G123WalkAgentCfg(walk_cfg.TienKungWalkAgentCfg):

    def __post_init__(self):
        self.save_interval = 1000
        self.amp_motion_files = [
            "legged_lab/envs/tienkung/datasets/motion_amp_expert/g1_23dof_s1_walking2_stageii.txt",
            "legged_lab/envs/tienkung/datasets/motion_amp_expert/g1_23dof_s3_walking1_stageii.txt",
            "legged_lab/envs/tienkung/datasets/motion_amp_expert/g1_23dof_s5_walking2_stageii.txt"
            ]
        self.experiment_name = "g1_23dof_walk"


@configclass
class G123WalkAgentCfgLafan(walk_cfg.TienKungWalkAgentCfg):

    def __post_init__(self):
        self.save_interval = 1000
        self.amp_motion_files = [
            "legged_lab/envs/tienkung/datasets/motion_amp_expert/g1_23dof_lafan1_walk1_subject5.txt"
            ]
        self.experiment_name = "walk_g123dof_afan"
