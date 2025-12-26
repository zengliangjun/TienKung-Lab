# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import argparse
import os

import torch
from isaaclab.app import AppLauncher

import os.path as osp
root = osp.join(osp.dirname(__file__), "../..")
import sys
if root not in sys.path:
    sys.path.insert(0, root)
rsl_rl_root = osp.join(root, "rsl_rl")
if rsl_rl_root not in sys.path:
    sys.path.insert(0, rsl_rl_root)

from legged_lab.utils import task_registry
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

args_cli.task = "walk_g123dof"
args_cli.headless = True
args_cli.load_run = "2025-12-22_15-54-18"
args_cli.num_envs = 1

# Start camera rendering
if "sensor" in args_cli.task:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg


def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.ranges.lin_vel_x = (1.0, 1.0)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)

    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"

    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 5
        env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        env_cfg.scene.terrain_generator.difficulty_range = (0.4, 0.4)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    runner_class: OnPolicyRunner | AmpOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    policy = runner.get_inference_policy(device=env.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard

        keyboard = Keyboard(env)  # noqa:F841

    obs, _ = env.get_observations()

    obs_test = torch.reshape(obs, (obs.shape[0], env.cfg.robot.actor_obs_history_length, -1))
    from isaaclab.assets.articulation import Articulation
    robot: Articulation = env.robot
    dof_pos = robot.data.default_joint_pos
    kps = robot.data.joint_stiffness
    kds = robot.data.joint_damping
    pos_limits = robot.data.joint_pos_limits
    effort_limits = robot.data.joint_effort_limits

    print("dof_pos", dof_pos)
    print("kps", kps)
    print("kds", kds)
    print("pos_limits", pos_limits)
    print("effort_limits", effort_limits)

    def position_control(action):
        """
        Apply position control using scaled action.

        Returns:
            np.ndarray: Target joint positions in MuJoCo order.
        """
        actions_scaled = action * env.cfg.robot.action_scale
        target_pos = actions_scaled + dof_pos
        clip_pos = torch.clip(target_pos, pos_limits[:, :, 0], pos_limits[:, :, 1])
        diff_pos = target_pos - clip_pos
        print("diff pos", diff_pos)
        return clip_pos


    def pd_control(target_q, q, kp, target_dq, dq, kd):
        return (target_q - q) * kp + (target_dq - dq) * kd

    import pickle
    count = 0
    while simulation_app.is_running():

        with torch.inference_mode():
            actions = policy(obs)
            pos = robot.data.joint_pos.clone()
            vel = robot.data.joint_vel.clone()

            target_dof_pos = position_control(actions)
            pd_torque = pd_control(target_dof_pos, pos, kps, torch.zeros_like(kds), vel, kds)

            obs_new, _, _, _ = env.step(actions)

            torque = robot.data.applied_torque
            print(pd_torque - torque)

            with open(f"test_data/test_{count:04d}.pkl", "w+b") as fd:
                items = {
                    "obs": obs[0].detach().cpu().numpy(),
                    "actions": actions[0].detach().cpu().numpy(),
                    "pos": pos[0].detach().cpu().numpy(),
                    "vel": vel[0].detach().cpu().numpy(),
                    "torque": torque[0].detach().cpu().numpy()
                }
                pickle.dump(items, fd)

            obs = obs_new

        count += 1
        if count > 100:
            break


if __name__ == "__main__":
    play()
    simulation_app.close()
