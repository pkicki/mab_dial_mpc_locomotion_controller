import os
import numpy as np
import jax
import jax.numpy as jnp
import yaml
import brax.envs as brax_envs

import dial_mpc.envs as dial_envs
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict
from dial_mpc.examples import examples
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.core.dial_core import MBDPI


class SBDialMPCLocomotionController:
    def __init__(self, real_robot: bool, tuda_robot: bool, spine_locked: bool):
        self.real_robot = real_robot
        self.tuda_robot = tuda_robot
        # TODO spine locked models are not well tested, do it before using them
        self.spine_locked = spine_locked

        self.num_joints = 13

        self.joint_positions = np.zeros(self.num_joints)
        self.joint_velocities = np.zeros(self.num_joints)
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.angular_velocity = np.zeros(3)
        self.x_goal_velocity = 0.0
        self.y_goal_velocity = 0.0
        self.yaw_goal_velocity = 0.0

        self.nominal_joint_positions = np.array([
            -0.1, 0.8, -1.5,
            0.1, -0.8, 1.5,
            -0.1, -1.0, 1.5,
            0.1, 1.0, -1.5,
            0.0
        ])
        self.max_joint_velocities = np.array([
            25.0, 25.0, 25.0,
            25.0, 25.0, 25.0,
            25.0, 25.0, 25.0,
            25.0, 25.0, 25.0,
            3.15
        ])

        self.mask_from_real_to_obssim = [3, 4, 5, 0, 1, 2, 6, 7, 8, 9, 10, 11, 12]
        self.mask_from_xmlsim_to_real = np.array([7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 0])
        self.mask_from_xmlsim_to_obssim = [10, 11, 12, 7, 8, 9, 1, 2, 3, 4, 5, 6, 0]

        self.config_path = os.path.join(os.path.dirname(__file__),
                                        "dial-mpc/dial_mpc/examples/mab_sb_trot.yaml")
        self.cli_args = type('CLIArgs', (object,), {'lporder': 3, 'lpfreq': 3,
                                                    'beta': None, 'hnode': 16})()
        self.controls, self.dial_mpc, self.state, self.rng = self.load_dial_mpc()
        self._it = 0
        self.trajectory_diffuse_factor = 0.5
        self.Ndiffuse = 2
        self.Ndiffuse_init = 10
        print("CONTROLLER READY")

    def load_dial_mpc(self):
        config_dict = yaml.safe_load(open(self.config_path))

        dial_config = load_dataclass_from_dict(DialConfig, config_dict)
        if self.cli_args.hnode is not None:
            dial_config.Hnode = self.cli_args.hnode
        rng = jax.random.PRNGKey(seed=0)

        # find env config
        env_config_type = dial_envs.get_config(dial_config.env_name)
        env_config = load_dataclass_from_dict(
            env_config_type, config_dict, convert_list_to_array=True
        )

        print("Creating environment")
        env = brax_envs.get_environment(dial_config.env_name, config=env_config)
        reset_env = jax.jit(env.reset)
        step_env = jax.jit(env.step)
        mbdpi = MBDPI(self.cli_args, dial_config, env)

        rng, rng_reset = jax.random.split(rng)
        state_init = reset_env(rng_reset)

        Y = jnp.ones([dial_config.Hnode + 1, mbdpi.nu]) * env.default_action

        return Y, mbdpi, state_init, rng

            
    def set_robot_internal_state(self, joint_positions: np.array, joint_velocities: np.array):
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities
        if self.real_robot and self.tuda_robot:
            self.joint_positions[-1] *= -1.0
            self.joint_velocities[-1] *= -1.0


    def set_robot_external_state(self, position: np.array, velocity: np.array,
                                 orientation: np.array, angular_velocity: np.array):
        self.position = position
        self.velocity = velocity
        self.orientation = orientation
        self.angular_velocity = angular_velocity


    def set_velocity_command(self, vx, vy, wz):
        self.x_goal_velocity = vx
        self.y_goal_velocity = vy
        self.yaw_goal_velocity = wz 

    def update_mjx_state(self):
        q = np.concatenate([self.position, self.orientation, self.joint_positions])
        qd = np.concatenate([self.velocity, self.angular_velocity, self.joint_velocities])
        pipeline_state = self.state.pipeline_state.replace(qpos=q, qvel=qd)
        step = self._it
        info = self.state.info
        info["step"] = step
        self.state = self.state.replace(pipeline_state=pipeline_state, info=info)

    def compute_target_joint_positions(self):
        def reverse_scan(rng_Y0_state, factor):
            rng, Y0, state = rng_Y0_state
            rng, Y0, info = self.dial_mpc.reverse_once(state, rng, Y0, factor)
            return (rng, Y0, state), info

        self.update_mjx_state()

        n_diffuse = self.Ndiffuse
        if self._it == 0:
            n_diffuse = self.Ndiffuse_init
            print("Performing JIT on DIAL-MPC")

        traj_diffuse_factors = (
            self.dial_mpc.sigma_control * self.trajectory_diffuse_factor ** (jnp.arange(n_diffuse))[:, None]
        )
        (self.rng, self.controls, _), info = jax.lax.scan(
            reverse_scan, (self.rng, self.controls, self.state), traj_diffuse_factors
        )

        action = self.controls[0]
        if self.spine_locked:
            action = np.insert(action, 0, 0.0)

        robot_action = action[self.mask_from_xmlsim_to_real] # TODO check if correct

        target_joint_positions = self.nominal_joint_positions + robot_action
        if self.real_robot and self.tuda_robot:
            target_joint_positions[-1] *= -1.0

        # update Y0
        self.controls = self.dial_mpc.shift(self.controls)
        self._it += 1

        return target_joint_positions


if __name__ == "__main__":
    SBDialMPCLocomotionController(
        real_robot=True,
        tuda_robot=True,
        spine_locked=True,
    )
