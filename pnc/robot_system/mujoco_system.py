import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)

from collections import OrderedDict

import mujoco
import numpy as np

from .base import RobotSystem
from util import geom


class MujocoRobotSystem(RobotSystem):
    def __init__(
        self, description_file, package_name, b_fixed_base, b_print_info=False
    ):
        super(MujocoRobotSystem, self).__init__(
            description_file, package_name, b_fixed_base, b_print_info
        )

    def _config_robot(self, description_file, package_name):
        self._model = mujoco.MjModel.from_xml_path(description_file)
        self._data = mujoco.MjData(self._model)

        # Set joint limits #
        # Position
        lower_jnt_pos_limit = np.copy(self._model.jnt_range[:, 0])
        upper_jnt_pos_limit = np.copy(self._model.jnt_range[:, 1])
        self._joint_pos_limit = np.stack(
            [lower_jnt_pos_limit[1:], upper_jnt_pos_limit[1:]], axis=1
        )

        # Velocity TODO: Update
        self._joint_vel_limit = np.ones_like(self._joint_pos_limit) * 100

        # Torque
        self._joint_trq_limit = self._model.actuator_forcerange

    def get_joint_idx(self, joint_name):
        if type(joint_name) is list:
            return [self.get_joint_idx(j_name) for j_name in joint_name]
        else:
            return self._model.body_jntadr[
                mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            ]

    def get_q_idx(self, joint_name):
        return self.get_joint_idx(joint_name)

    def get_q_dot_idx(self, joint_name):
        return self.get_joint_idx(joint_name)

    def create_cmd_ordered_dict(self, joint_pos_cmd, joint_vel_cmd, joint_trq_cmd):
        command = OrderedDict()
        command["joint_pos"] = OrderedDict()
        command["joint_vel"] = OrderedDict()
        command["joint_trq"] = OrderedDict()

        for k, v in self._joint_id.items():
            command["joint_pos"][k] = joint_pos_cmd[v]
            command["joint_vel"][k] = joint_vel_cmd[v]
            command["joint_trq"][k] = joint_trq_cmd[v]

        return command

    def update_system(
        self,
        base_com_pos,
        base_com_quat,
        base_com_lin_vel,
        base_com_ang_vel,
        base_joint_pos,
        base_joint_quat,
        base_joint_lin_vel,
        base_joint_ang_vel,
        joint_pos,
        joint_vel,
        b_cent=False,
    ):
        # assert len(joint_pos.keys()) == self._n_a

        # if not self._b_fixed_base:
        #     # Floating Based Robot
        #     self._q[0:3] = np.copy(base_joint_pos)
        #     self._q[3:7] = np.copy(base_joint_quat)

        #     rot_w_basejoint = geom.quat_to_rot(base_joint_quat)
        #     twist_basejoint_in_world = np.zeros(6)
        #     twist_basejoint_in_world[0:3] = base_joint_ang_vel
        #     twist_basejoint_in_world[3:6] = base_joint_lin_vel
        #     augrot_joint_world = np.zeros((6, 6))
        #     augrot_joint_world[0:3, 0:3] = rot_w_basejoint.transpose()
        #     augrot_joint_world[3:6, 3:6] = rot_w_basejoint.transpose()
        #     twist_basejoint_in_joint = np.dot(augrot_joint_world,
        #                                       twist_basejoint_in_world)
        #     self._q_dot[0:3] = twist_basejoint_in_joint[3:6]
        #     self._q_dot[3:6] = twist_basejoint_in_joint[0:3]

        mujoco.mj_kinematics(self._model, self._data)

        if b_cent:
            self._update_centroidal_quantities()

    def _update_centroidal_quantities(self):
        pin.ccrba(self._model, self._data, self._q, self._q_dot)

        self._hg = np.zeros_like(self._data.hg)
        self._hg[0:3] = np.copy(self._data.hg.angular)
        self._hg[3:6] = np.copy(self._data.hg.linear)

        self._Ag = np.zeros_like(self._data.Ag)
        self._Ag[0:3] = np.copy(self._data.Ag[3:6, :])
        self._Ag[3:6] = np.copy(self._data.Ag[0:3, :])

        self._Ig = np.zeros_like(self._data.Ig)
        self._Ig[0:3, 0:3] = np.copy(self._data.Ig)[3:6, 3:6]
        self._Ig[3:6, 3:6] = np.copy(self._data.Ig)[0:3, 0:3]

    def get_q(self):
        return np.copy(self._data.qpos)

    def get_q_dot(self):
        return np.copy(self._data.qvel)

    def get_mass_matrix(self):
        return np.copy(
            mujoco.mj_fullM(
                self._model, np.zeros((self._model.nv, self._model.nv)), self._data.qM
            )
        )

    def get_gravity(self):
        return np.copy(self._data.qfrc_bias)

    def get_coriolis(self):
        result = np.ndarray(self.n_q_dot, dtype=float)
        mujoco.mj_rne(self._model, self._data, 0, result)
        return result

    def get_com_pos(self):
        mujoco.mj_comPos(self._model, self._data)
        return np.copy(self._data.subtree_com[0])

    def get_com_lin_vel(self):
        mujoco.mj_comVel(self._model, self._data)
        return np.copy(self._data.cvel[0, 3:])

    def get_com_lin_jacobian(self):
        return np.copy(pin.jacobianCenterOfMass(self._model, self._data, self._q))

    def get_com_lin_jacobian_dot(self):
        return np.copy(
            (
                pin.computeCentroidalMapTimeVariation(
                    self._model, self._data, self._q, self._q_dot
                )[0:3, :]
            )
            / self._model.body_mass
        )

    def get_link_iso(self, link_id):
        print(link_id)
        frame_id = self._model.getFrameId(link_id)
        trans = pin.updateFramePlacement(self._model, self._data, frame_id)

        ret = np.eye(4)
        ret[0:3, 0:3] = trans.rotation
        ret[0:3, 3] = trans.translation
        return np.copy(ret)

    def get_link_vel(self, link_id):
        ret = np.zeros(6)
        frame_id = self._model.getFrameId(link_id)

        spatial_vel = pin.getFrameVelocity(
            self._model, self._data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        ret[0:3] = spatial_vel.angular
        ret[3:6] = spatial_vel.linear

        return np.copy(ret)

    def get_link_jacobian(self, link_id):
        frame_id = self._model.getFrameId(link_id)
        pin.computeJointJacobians(self._model, self._data, self._q)
        jac = pin.getFrameJacobian(
            self._model, self._data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        # Mujoco has linear on top of angular
        ret = np.zeros_like(jac)
        ret[0:3] = jac[3:6]
        ret[3:6] = jac[0:3]

        return np.copy(ret)

    def get_link_jacobian_dot_times_qdot(self, link_id):
        frame_id = self._model.getFrameId(link_id)

        pin.forwardKinematics(
            self._model, self._data, self._q, self._q_dot, 0 * self._q_dot
        )
        jdot_qdot = pin.getFrameClassicalAcceleration(
            self._model, self._data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        ret = np.zeros_like(jdot_qdot)
        ret[0:3] = jdot_qdot.angular
        ret[3:6] = jdot_qdot.linear

        return np.copy(ret)

    def get_Ag(self):
        return np.copy(self._Ag)

    def get_Ig(self):
        return np.copy(self._Ig)

    def get_hg(self):
        return np.copy(self._hg)

    # We assume that we are using a floating base robot
    # CHECKED
    @property
    def n_floating(self):
        return 6

    @property
    def n_q(self):
        return self._model.nq

    # CHECKED
    @property
    def n_q_dot(self):
        return self._model.nv

    # CHECKED
    @property
    def n_a(self):
        return self.n_q_dot - self.n_floating

    @property
    def total_mass(self):
        return np.sum(self._model.body_mass)

    @property
    def joint_positions(self):
        return self._model.qpos[1:]

    @property
    def joint_velocities(self):
        return self._model.qvel[1:]
