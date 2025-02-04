from time import perf_counter
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from nav_msgs.msg import Odometry

from hb40_commons.msg import BridgeData
from hb40_commons.msg import JointCommand
from mab_dial_mpc_locomotion_controller.sb_dial_mpc_locomotion_controller import SBDialMPCLocomotionController


class SBDialMPCLocomotionControllerNode(Node):
    def __init__(self):
        super().__init__("sb_dial_mpc_locomotion_controller")
        self.real_robot = True
        self.tuda_robot = False

        #self.spine_locked = True
        self.spine_locked = False

        self.locomotion_controller = SBDialMPCLocomotionController(
            real_robot=self.real_robot,
            tuda_robot=self.tuda_robot,
            spine_locked=self.spine_locked,
        )

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.bridge_data_subscription = self.create_subscription(
            BridgeData,
            "/hb40/bridge_data",
            self.bridge_data_callback,
            qos_profile=qos_profile)

        self.optitrack_subscription = self.create_subscription(
            Odometry,
            "/optitrack/odom",
            self.optitrack_callback,
            qos_profile=qos_profile)
        
        self.velocity_command_subscription = self.create_subscription(
            Twist,
            "/hb40/velocity_command",
            self.velocity_command_callback,
            qos_profile=qos_profile)
        
        self.joystick_subscription = self.create_subscription(
            Joy,
            "/hb40/joy",
            self.joystick_callback,
            qos_profile=qos_profile)
        
        self.joint_commands_publisher = self.create_publisher(
            JointCommand,
            "/hb40/joint_command_" if self.real_robot else "/hb40/joint_commandHighPrio",
            qos_profile=qos_profile)
        
        self.mpc_active = False
        timer_period = 0.05  # 20 Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.kp = [20.0,] * 13
        self.kd = [0.5,] * 13
        if self.spine_locked:
            self.kp[-1] = 0.0
            self.kd[-1] = 0.0

    def bridge_data_callback(self, msg):
        self.locomotion_controller.set_robot_internal_state(
            joint_positions=np.array(msg.joint_position),
            joint_velocities=np.array(msg.joint_velocity),
        )

    def optitrack_callback(self, msg):
        self.locomotion_controller.set_robot_external_state(
            position=np.array([msg.pose.pose.position.x,
                               msg.pose.pose.position.y,
                               msg.pose.pose.position.z]),
            velocity=np.array([msg.twist.twist.linear.x,
                               msg.twist.twist.linear.y,
                               msg.twist.twist.linear.z]),
            orientation=np.array([msg.pose.pose.orientation.w,
                                  msg.pose.pose.orientation.x,
                                  msg.pose.pose.orientation.y,
                                  msg.pose.pose.orientation.z]),
            angular_velocity=np.array([msg.twist.twist.angular.x,
                                       msg.twist.twist.angular.y,
                                       msg.twist.twist.angular.z]),
        )

    def velocity_command_callback(self, msg):
        pass
        # TODO: we need to pass these commands to the reward function one day
        #self.locomotion_controller.set_velocity_command(msg.linear.x,
        #                                                msg.linear.y,
        #                                                msg.angular.z)


    def joystick_callback(self, msg):
        if msg.buttons[0] == 1 and self.mpc_active:
            self.mpc_active = False
        elif msg.buttons[1] == 1 and not self.mpc_active:
            self.mpc_active = True


    def timer_callback(self):
        if not self.mpc_active:
            return

        print("CALL MPC")
        t0 = perf_counter()
        target_joint_positions = self.locomotion_controller.compute_target_joint_positions()
        t1 = perf_counter()
        print("MPC took", t1 - t0, "seconds")
        self.publish_joint_command(target_joint_positions)


    def publish_joint_command(self, target_joint_positions):
        joint_command_msg = JointCommand()
        joint_command_msg.header.stamp = self.get_clock().now().to_msg()
        joint_command_msg.source_node = "nn_controller" # TODO change to real node name if possible
        joint_command_msg.name = [""] * 13
        joint_command_msg.kp = self.kp
        joint_command_msg.kd = self.kd
        joint_command_msg.t_pos = target_joint_positions.tolist()
        joint_command_msg.t_vel = np.zeros_like(target_joint_positions).tolist()
        joint_command_msg.t_trq = np.zeros_like(target_joint_positions).tolist()

        self.joint_commands_publisher.publish(joint_command_msg)


def main(args=None):
    rclpy.init(args=args)
    robot_handler = SBDialMPCLocomotionControllerNode()
    rclpy.spin(robot_handler)
    robot_handler.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
