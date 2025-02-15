from copy import copy
import os
import numpy as np
import jax
import jax.numpy as jnp
import yaml
import brax.envs as brax_envs
from time import perf_counter

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
        self.position = np.array([0., 0., 0.3])
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self.velocity = np.zeros(3)
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
        self.cli_args = type('CLIArgs', (object,), {
            'noise_type': "lp",
            'lporder': 3, 'lpfreq': 3,
            'beta': 3, 'hnode': 12, 'hsample': 12})()
        self.controls, self.dial_mpc, self.state, self.rng = self.load_dial_mpc()

        self.ros2xml = None
        self.xml2ros = None

        self._it = 0
        self.trajectory_diffuse_factor = 0.5
        self.Ndiffuse = 2
        self.Ndiffuse_init = 10
        def reverse_scan(rng_Y0_state, factor):
            rng, Y0, state = rng_Y0_state
            rng, Y0, info = self.dial_mpc.reverse_once(state, rng, Y0, factor)
            return (rng, Y0, state), info
        self.reverse_scan = jax.jit(reverse_scan)
        print("CONTROLLER READY")

    def load_dial_mpc(self):
        config_dict = yaml.safe_load(open(self.config_path))

        dial_config = load_dataclass_from_dict(DialConfig, config_dict)
        if self.cli_args.hnode is not None:
            dial_config.Hnode = self.cli_args.hnode
        if self.cli_args.noise_type is not None:
            dial_config.noise_type = self.cli_args.noise_type
        if self.cli_args.hsample is not None:
            dial_config.hsample = self.cli_args.hsample
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

    def setup_ros_xml_mapping(self, joint_names):
        actadr = self.dial_mpc.env.sys.mj_model.name_actuatoradr
        xml_act_names = [self.dial_mpc.env.sys.mj_model.names[actadr[i]:].split(b'\x00', 1)[0].decode() for i in range(len(actadr))]
        xml_act_names = [x.lower()
                         .replace("hip", "j0")
                         .replace("thigh", "j1")
                         .replace("calf", "j2")
                         .replace("spine", "sp_j0") for x in xml_act_names]
        ros_names2xml_idx = {x: i for i, x in enumerate(xml_act_names)}

        self.xml2ros = np.array([ros_names2xml_idx[n] for n in joint_names])
        self.ros2xml = np.array([joint_names.index(n) for n in xml_act_names])


    def set_robot_internal_state(self, joint_names: list,
                                 joint_positions: np.array, joint_velocities: np.array):
        if self.ros2xml is None or self.xml2ros is None:
            self.setup_ros_xml_mapping(joint_names)
        self.joint_positions = joint_positions[self.ros2xml]
        self.joint_velocities = joint_velocities[self.ros2xml]
        if self.real_robot and self.tuda_robot:
            self.joint_positions[0] *= -1.0
            self.joint_velocities[0] *= -1.0


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
        self.update_mjx_state()

        n_diffuse = self.Ndiffuse
        if self._it == 0:
            n_diffuse = self.Ndiffuse_init
            print("Performing JIT on DIAL-MPC")

        traj_diffuse_factors = (
            self.dial_mpc.sigma_control * self.trajectory_diffuse_factor ** (jnp.arange(n_diffuse))[:, None]
        )
        (self.rng, self.controls, _), info = jax.lax.scan(
            self.reverse_scan, (self.rng, self.controls, self.state), traj_diffuse_factors
        )

        action = self.controls[0]
        target_joint_positions_xml = self.dial_mpc.env.act2joint(action)
        if self.spine_locked:
            action = np.insert(action, 0, 0.0)

        target_joint_positions_ros = target_joint_positions_xml[self.xml2ros] # TODO check if correct
        #target_joint_positions = action[self.mask_from_xmlsim_to_real] # TODO check if correct

        if self.real_robot and self.tuda_robot:
            target_joint_positions_ros[0] *= -1.0

        # update Y0
        self.controls = self.dial_mpc.shift(self.controls)
        self._it += 1

        return target_joint_positions_ros


if __name__ == "__main__":
    controller = SBDialMPCLocomotionController(
        real_robot=True,
        tuda_robot=True,
        spine_locked=True,
    )
    for i in range(10):
        t0 = perf_counter()
        controller.compute_target_joint_positions()
        t1 = perf_counter()
        print("MPC took", t1 - t0, "seconds")
