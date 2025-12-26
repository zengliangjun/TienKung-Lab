"""This script replay a motion from a csv file and output it to a npz file

.. code-block:: bash

    # Usage
    python csv_to_npz.py --input_file LAFAN/dance1_subject2.csv --input_fps 30 --frame_range 122 722 \
    --output_file ./motions/dance1_subject2.npz --output_fps 50
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np

from isaaclab.app import AppLauncher

import os.path as osp
root = osp.join(osp.dirname(__file__), "../..")
import sys
if root not in sys.path:
    sys.path.insert(0, root)
rsl_rl_root = osp.join(root, "rsl_rl")
if rsl_rl_root not in sys.path:
    sys.path.insert(0, rsl_rl_root)



# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
parser.add_argument("--input_file", type=str, required=True, help="The path to the input motion csv file.")
parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded."
    ),
)
parser.add_argument("--output_name", type=str, required=True, help="The name of the motion npz file.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp
from isaaclab.utils.math import quat_apply
from isaaclab.assets.articulation import Articulation

##
# Pre-defined configs
##

from legged_lab.envs.tienkung import g123dof_info
from legged_lab.assets.unitree import unitree_g123dof
import pickle


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = unitree_g123dof.UNITREE_G1_23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the csv file."""
        if self.frame_range is None:
            if self.motion_file.endswith("txt"):
                motion_numpy = np.loadtxt(self.motion_file, delimiter=",")
            elif self.motion_file.endswith("pkl"):
                with open(self.motion_file, "rb") as f:
                    motion_data = pickle.load(f)
                root_pos = motion_data["root_pos"]
                root_rot = motion_data["root_rot"]  #  [:, [3, 0, 1, 2]]  # xyzw → wxyz
                dof_pos = motion_data["dof_pos"]

                motion_numpy = np.concatenate((root_pos, root_rot, dof_pos), axis=-1)
        else:
            if self.motion_file.endswith("txt"):
                motion_numpy = np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            elif self.motion_file.endswith("pkl"):
                with open(self.motion_file, "rb") as f:
                    motion_data = pickle.load(f)
                root_pos = motion_data["root_pos"]
                root_rot = motion_data["root_rot"]  #  [:, [3, 0, 1, 2]]  # xyzw → wxyz
                dof_pos = motion_data["dof_pos"]

                motion_numpy = np.concatenate((root_pos, root_rot, dof_pos), axis=-1)
                motion_numpy = motion_numpy[self.frame_range[0] - 1: self.frame_range[1]]

        motion = torch.from_numpy(motion_numpy)
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # convert to wxyz
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Load motion
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
    )

    # Extract scene entities
    robot: Articulation = scene["robot"]
    mujoco_joint_ids = robot.find_joints(g123dof_info.mujoco_joint_names, preserve_order=True)[0]
    left_leg_ids = robot.find_joints(g123dof_info.mujoco_leftleg_names, preserve_order=True)[0]
    right_leg_ids = robot.find_joints(g123dof_info.mujoco_rightleg_names, preserve_order=True)[0]
    left_arm_ids = robot.find_joints(g123dof_info.mujoco_leftarm_names, preserve_order=True)[0]
    right_arm_ids = robot.find_joints(g123dof_info.mujoco_rightarm_names, preserve_order=True)[0]


    mujoco_left_leg_ids = []
    for name in g123dof_info.mujoco_leftleg_names:
        mujoco_left_leg_ids.append(g123dof_info.mujoco_joint_names.index(name))

    mujoco_right_leg_ids = []
    for name in g123dof_info.mujoco_rightleg_names:
        mujoco_right_leg_ids.append(g123dof_info.mujoco_joint_names.index(name))


    mujoco_left_arm_ids = []
    for name in g123dof_info.mujoco_leftarm_names:
        mujoco_left_arm_ids.append(g123dof_info.mujoco_joint_names.index(name))

    mujoco_right_arm_ids = []
    for name in g123dof_info.mujoco_rightarm_names:
        mujoco_right_arm_ids.append(g123dof_info.mujoco_joint_names.index(name))


    init_leftarm_names = [
        'left_shoulder_pitch_joint',
        'left_shoulder_roll_joint',
        #'left_shoulder_yaw_joint',
        'left_elbow_joint'
    ]
    init_rightarm_names = [
        'right_shoulder_pitch_joint',
        'right_shoulder_roll_joint',
        #'right_shoulder_yaw_joint',
        'right_elbow_joint'
    ]
    init_left_arm_ids = robot.find_joints(init_leftarm_names, preserve_order=True)[0]
    init_right_arm_ids = robot.find_joints(init_rightarm_names, preserve_order=True)[0]

    mujoco_init_left_arm_ids = []
    for name in init_leftarm_names:
        mujoco_init_left_arm_ids.append(g123dof_info.mujoco_joint_names.index(name))

    mujoco_init_right_arm_ids = []
    for name in init_rightarm_names:
        mujoco_init_right_arm_ids.append(g123dof_info.mujoco_joint_names.index(name))

    init_scales = torch.tensor([[0.85, 0.95, 0.8]], device=sim.device)

    left_shoulder_yaw_ids = robot.find_joints(['left_shoulder_yaw_joint'], preserve_order=True)[0]
    right_shoulder_yaw_ids = robot.find_joints(['right_shoulder_yaw_joint'], preserve_order=True)[0]

    mujoco_left_shoulder_yaw_ids = [g123dof_info.mujoco_joint_names.index('left_shoulder_yaw_joint')]
    mujoco_right_shoulder_yaw_ids = [g123dof_info.mujoco_joint_names.index('right_shoulder_yaw_joint')]


    feet_body_ids, _ = robot.find_bodies(
        name_keys=g123dof_info.feet_body_names, preserve_order=True
    )
    elbow_body_ids, _ = robot.find_bodies(
        name_keys=g123dof_info.elbow_body_names, preserve_order=True
    )

    left_arm_local_vec = torch.tensor([[0.0, 0.0, -0.3]]).to(robot.device)
    right_arm_local_vec = torch.tensor([[0.0, 0.0, -0.3]]).to(robot.device)

    # ------- data logger -------------------------------------------------------
    file_saved = False
    # --------------------------------------------------------------------------
    all_frames = []
    # Simulation loop
    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        # set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        if False:
            joint_pos[:, mujoco_joint_ids] = motion_dof_pos
            joint_vel[:, mujoco_joint_ids] = motion_dof_vel
        else:
            joint_pos[:, left_leg_ids] = motion_dof_pos[:, mujoco_left_leg_ids]
            joint_pos[:, right_leg_ids] = motion_dof_pos[:, mujoco_right_leg_ids]
            joint_pos[:, init_left_arm_ids] = motion_dof_pos[:, mujoco_init_left_arm_ids] * init_scales
            joint_pos[:, init_right_arm_ids] = motion_dof_pos[:, mujoco_init_right_arm_ids] * init_scales

            joint_vel[:, left_leg_ids] = motion_dof_vel[:, mujoco_left_leg_ids]
            joint_vel[:, right_leg_ids] = motion_dof_vel[:, mujoco_right_leg_ids]
            joint_vel[:, init_left_arm_ids] = motion_dof_vel[:, mujoco_init_left_arm_ids]
            joint_vel[:, init_right_arm_ids] = motion_dof_vel[:, mujoco_init_right_arm_ids]

        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            left_hand_pos = (
                robot.data.body_state_w[:, elbow_body_ids[0], :3]
                - robot.data.root_state_w[:, 0:3]
                + quat_apply(robot.data.body_state_w[:, elbow_body_ids[0], 3:7], left_arm_local_vec)
            )
            right_hand_pos = (
                robot.data.body_state_w[:, elbow_body_ids[1], :3]
                - robot.data.root_state_w[:, 0:3]
                + quat_apply(robot.data.body_state_w[:, elbow_body_ids[1], 3:7], right_arm_local_vec)
            )
            left_hand_pos = quat_apply(quat_conjugate(robot.data.root_state_w[:, 3:7]), left_hand_pos)
            right_hand_pos = quat_apply(quat_conjugate(robot.data.root_state_w[:, 3:7]), right_hand_pos)
            left_foot_pos = (
                robot.data.body_state_w[:, feet_body_ids[0], :3] - robot.data.root_state_w[:, 0:3]
            )
            right_foot_pos = (
                robot.data.body_state_w[:, feet_body_ids[1], :3] - robot.data.root_state_w[:, 0:3]
            )
            left_foot_pos = quat_apply(quat_conjugate(robot.data.root_state_w[:, 3:7]), left_foot_pos)
            right_foot_pos = quat_apply(quat_conjugate(robot.data.root_state_w[:, 3:7]), right_foot_pos)

            left_leg_dof_pos =  robot.data.joint_pos[:, left_leg_ids]
            right_leg_dof_pos = robot.data.joint_pos[:, right_leg_ids]
            left_leg_dof_vel =  robot.data.joint_vel[:, left_leg_ids]
            right_leg_dof_vel = robot.data.joint_vel[:, right_leg_ids]
            left_arm_dof_pos =  robot.data.joint_pos[:, left_arm_ids]
            right_arm_dof_pos = robot.data.joint_pos[:, right_arm_ids]
            left_arm_dof_vel =  robot.data.joint_vel[:, left_arm_ids]
            right_arm_dof_vel = robot.data.joint_vel[:, right_arm_ids]

            frame = torch.cat(
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
                        right_foot_pos
                    ),
                    dim=-1,
                ).cpu().numpy().reshape(-1)
            all_frames.append(frame)

        if reset_flag and not file_saved:
            file_saved = True
            all_frames = all_frames[250: -250]
            all_frames_np = np.stack(all_frames, axis=0)
            np.savetxt(args_cli.output_name, all_frames_np, fmt='%f', delimiter=', ')

            with open(args_cli.output_name, 'r') as f:
                frames_data = f.readlines()

            frames_data_len = len(frames_data)
            with open(args_cli.output_name, 'w') as f:
                f.write('{\n')
                f.write('"LoopMode": "Wrap",\n')
                f.write(f'"FrameDuration": {1.0 / args_cli.output_fps:.3f},\n')
                f.write('"EnableCycleOffsetPosition": true,\n')
                f.write('"EnableCycleOffsetRotation": true,\n')
                f.write('"MotionWeight": 0.5,\n\n')
                f.write('"Frames":\n[\n')

                for i, line in enumerate(frames_data):
                    line_start_str = '  ['
                    if i == frames_data_len - 1:
                        f.write(line_start_str + line.rstrip() + ']\n')
                    else:
                        f.write(line_start_str + line.rstrip() + '],\n')

                f.write(']\n}')

            print(f"✅ Successfully converted to {args_cli.output_name}")
            return

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(
        sim,
        scene
    )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
