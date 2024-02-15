import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
import copy

from pnc.interface import Interface
from pnc.gr1_pnc.interrupt_logic import GR1ManipulationInterruptLogic
from pnc.gr1_pnc.state_provider import GR1ManipulationStateProvider
from pnc.gr1_pnc.state_estimator import GR1ManipulationStateEstimator
from pnc.gr1_pnc.control_architecture import GR1ManipulationControlArchitecture
from pnc.data_saver import DataSaver
from pnc.robot_system import PinocchioRobotSystem


class GR1ManipulationInterface(Interface):
    def __init__(self, path_to_model, config):
        super(GR1ManipulationInterface, self).__init__()

        self._robot = PinocchioRobotSystem(
            path_to_model + "/GR1T1.urdf", path_to_model, False, False
        )

        self._config = config
        self._sp = GR1ManipulationStateProvider(self._robot)
        self._se = GR1ManipulationStateEstimator(self._robot, self._config)
        self._control_architecture = GR1ManipulationControlArchitecture(
            self._robot, self._config
        )
        self._interrupt_logic = GR1ManipulationInterruptLogic(
            self._control_architecture
        )
        if self._config["Simulation"]["Save Data"]:
            self._data_saver = DataSaver()
            self._data_saver.add("joint_pos_limit", self._robot.joint_pos_limit)
            self._data_saver.add("joint_vel_limit", self._robot.joint_vel_limit)
            self._data_saver.add("joint_trq_limit", self._robot.joint_trq_limit)

    def get_command(self, sensor_data):
        if self._config["Simulation"]["Save Data"]:
            self._data_saver.add("time", self._running_time)
            self._data_saver.add("phase", self._control_architecture.state)

        # Update State Estimator
        if self._count == 0:
            self._se.initialize(sensor_data)
        self._se.update(sensor_data)

        # Process Interrupt Logic
        self._interrupt_logic.process_interrupts()

        # Compute Cmd
        command = self._control_architecture.get_command()

        if self._config["Simulation"]["Save Data"] and (
            self._count % self._config["Simulation"]["Save Frequency"] == 0
        ):
            self._data_saver.add("joint_pos", self._robot.joint_positions)
            self._data_saver.add("joint_vel", self._robot.joint_velocities)
            self._data_saver.advance()

        # Increase time variables
        self._count += 1
        self._running_time += self._config["Simulation"]["Control Period"]
        self._sp.curr_time = self._running_time
        self._sp.prev_state = self._control_architecture.prev_state
        self._sp.state = self._control_architecture.state

        return copy.deepcopy(command)

    @property
    def interrupt_logic(self):
        return self._interrupt_logic

    @property
    def config(self):
        return self._config
