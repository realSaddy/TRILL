import numpy as np

# from pnc.gr1_pnc.rolling_joint_constraint import (
#     GR1ManipulationRollingJointConstraint,
# )
from pnc.wbc.basic_contact import SurfaceContact
from pnc.wbc.basic_task import BasicTask
from pnc.wbc.tci_container import TCIContainer
import ipdb

class GR1ManipulationTCIContainer(TCIContainer):
    def __init__(self, robot, config):
        super(GR1ManipulationTCIContainer, self).__init__(robot)

        self._robot = robot
        self._config = config

        save_data = config["Simulation"]["Save Data"]
        hierarchy_config = config["Whole-Body Contol"]["Hierarchy"]
        kp_config = config["Whole-Body Contol"]["kp"]
        kd_config = config["Whole-Body Contol"]["kd"]

        # ======================================================================
        # Initialize Task
        # ======================================================================
        # COM Task
        self._com_task = BasicTask(robot, "COM", 3, "com", save_data)
        # self._cam_task = BasicTask(robot, "CAM", 3, 'com', save_data)

        # Torso orientation task
        self._torso_ori_task = BasicTask(
            robot, "LINK_ORI", 3, "torso_com_link", save_data
        )

        # Upperbody joints
        upperbody_joint = [
            "l_shoulder_pitch",
            "l_shoulder_roll",
            "l_shoulder_yaw",
            "l_elbow_pitch",
            "l_wrist_yaw",
            "l_wrist_roll",
            "l_wrist_pitch",
            "r_shoulder_pitch",
            "r_shoulder_roll",
            "r_shoulder_yaw",
            "r_elbow_pitch",
            "r_wrist_yaw",
            "r_wrist_roll",
            "r_wrist_pitch",
            "head_yaw",
            "head_pitch",
            "head_roll",
            "waist_yaw",
            "waist_pitch",
            "waist_roll"
            
        ]
        self._upper_body_task = BasicTask(
            robot, "SELECTED_JOINT", len(upperbody_joint), upperbody_joint, save_data
        )
        # Lhand Pos Task
        self._lhand_pos_task = BasicTask(
            robot, "LINK_XYZ", 3, "left_hand_link", save_data
        )
        # Rhand Pos Task
        self._rhand_pos_task = BasicTask(
            robot, "LINK_XYZ", 3, "right_hand_link", save_data
        )
        # Lhand Ori Task
        self._lhand_ori_task = BasicTask(
            robot, "LINK_ORI", 3, "left_hand_link", save_data
        )
        # Rhand Ori Task
        self._rhand_ori_task = BasicTask(
            robot, "LINK_ORI", 3, "right_hand_link", save_data
        )
        # Rfoot Pos Task
        self._rfoot_pos_task = BasicTask(
            robot, "LINK_XYZ", 3, "right_foot_link", save_data
        )
        # Lfoot Pos Task
        self._lfoot_pos_task = BasicTask(
            robot, "LINK_XYZ", 3, "left_foot_link", save_data
        )
        # Rfoot Ori Task
        self._rfoot_ori_task = BasicTask(
            robot, "LINK_ORI", 3, "right_foot_link", save_data
        )
        # Lfoot Ori Task
        self._lfoot_ori_task = BasicTask(
            robot, "LINK_ORI", 3, "left_foot_link", save_data
        )

        self._task_list = [
            self._com_task,
            # self._cam_task,
            self._torso_ori_task,
            self._upper_body_task,
            self._lhand_pos_task,
            self._rhand_pos_task,
            self._lhand_ori_task,
            self._rhand_ori_task,
            self._rfoot_pos_task,
            self._lfoot_pos_task,
            self._rfoot_ori_task,
            self._lfoot_ori_task,
        ]

        hierachy_keys = [
            "COM",
            # 'CAM',
            "Torso",
            "Upper Body",
            "Hand Pos Min",
            "Hand Pos Min",
            "Hand Quat Min",
            "Hand Quat Min",
            "Contact Foot",
            "Contact Foot",
            "Contact Foot",
            "Contact Foot",
        ]

        gain_keys = [
            "COM",
            # 'CAM',
            "Torso",
            "Upper Body",
            "Hand Pos",
            "Hand Pos",
            "Hand Quat",
            "Hand Quat",
            "Foot Pos",
            "Foot Pos",
            "Foot Quat",
            "Foot Quat",
        ]

        for task, hiearchy_key, gain_key in zip(
            self._task_list, hierachy_keys, gain_keys
        ):
            # ipdb.set_trace()
            task.kp = np.array(kp_config[gain_key])
            task.kd = np.array(kd_config[gain_key])
            task.w_hierarchy = hierarchy_config[hiearchy_key]

        # ======================================================================
        # Initialize Contact
        # ======================================================================
        # Rfoot Contact
        self._rfoot_contact = SurfaceContact(
            robot, "right_foot_link", 0.115, 0.065, 0.3, save_data
        )
        self._rfoot_contact.rf_z_max = 1e-3  # Initial rf_z_max
        # Lfoot Contact
        self._lfoot_contact = SurfaceContact(
            robot, "left_foot_link", 0.115, 0.065, 0.3, save_data
        )
        self._lfoot_contact.rf_z_max = 1e-3  # Initial rf_z_max

        self._contact_list = [self._rfoot_contact, self._lfoot_contact]

        # ======================================================================
        # Initialize Internal Constraint
        # ======================================================================
        # self._rolling_joint_constraint = GR1ManipulationRollingJointConstraint(robot)
        # self._internal_constraint_list = [self._rolling_joint_constraint]
        self._internal_constraint_list = []

    @property
    def com_task(self):
        return self._com_task

    @property
    def cam_task(self):
        return self._cam_task

    @property
    def torso_ori_task(self):
        return self._torso_ori_task

    @property
    def upper_body_task(self):
        return self._upper_body_task

    @property
    def rfoot_pos_task(self):
        return self._rfoot_pos_task

    @property
    def lfoot_pos_task(self):
        return self._lfoot_pos_task

    @property
    def rfoot_ori_task(self):
        return self._rfoot_ori_task

    @property
    def lfoot_ori_task(self):
        return self._lfoot_ori_task

    @property
    def rhand_pos_task(self):
        return self._rhand_pos_task

    @property
    def lhand_pos_task(self):
        return self._lhand_pos_task

    @property
    def rhand_ori_task(self):
        return self._rhand_ori_task

    @property
    def lhand_ori_task(self):
        return self._lhand_ori_task

    @property
    def rfoot_contact(self):
        return self._rfoot_contact

    @property
    def lfoot_contact(self):
        return self._lfoot_contact

    @property
    def task_list(self):
        return self._task_list

    @property
    def contact_list(self):
        return self._contact_list

    @property
    def internal_constraint_list(self):
        return self._internal_constraint_list
