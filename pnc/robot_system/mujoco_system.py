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

        lower_jnt_pos_limit = np.copy(self._model.jnt_range[:, 0])
        upper_jnt_pos_limit = np.copy(self._model.jnt_range[:, 1])
        self._joint_pos_limit = np.stack(
            [lower_jnt_pos_limit[1:], upper_jnt_pos_limit[1:]], axis=1
        )
        self._joint_vel_limit = np.ones_like(self._joint_pos_limit) * 100
        self._joint_trq_limit = self._model.actuator_forcerange

    def get_joint_idx(self, joint_name):
        if type(joint_name) is list:
            return [self.get_joint_idx(j_name) for j_name in joint_name]
        else:
            return (
                self._model.jnt_dofadr[
                    mujoco.mj_name2id(
                        self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                    )
                ]
                - self.n_floating
            )

    def get_body_idx(self, body_name):
        if type(body_name) is list:
            return [self.get_body_idx(b_name) for b_name in body_name]
        else:
            return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    def get_q_idx(self, joint_name):
        if type(joint_name) is list:
            return [self.get_q_idx(j_name) for j_name in joint_name]
        else:
            return self._model.jnt_qposadr[
                mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            ]

    def get_q_dot_idx(self, joint_name):
        if type(joint_name) is list:
            return [self.get_q_dot_idx(j_name) for j_name in joint_name]
        else:
            return self._model.jnt_dofadr[
                mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            ]

    def create_cmd_ordered_dict(self, joint_pos_cmd, joint_vel_cmd, joint_trq_cmd):
        command = OrderedDict()
        command["joint_pos"] = OrderedDict()
        command["joint_vel"] = OrderedDict()
        command["joint_trq"] = OrderedDict()

        for i in range(self._model.njnt):
            pos = self._model.jnt_dofadr[i] - self.n_floating
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name == "root":
                continue

            command["joint_pos"][name] = joint_pos_cmd[pos]
            command["joint_vel"][name] = joint_vel_cmd[pos]
            command["joint_trq"][name] = joint_trq_cmd[pos]

        return command

    count = 0

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
        self._q = np.zeros(self.n_q)
        self._q_dot = np.zeros(self.n_q_dot)
        self._joint_positions = np.zeros(self.n_a)
        self._joint_velocities = np.zeros(self.n_a)
        if not self._b_fixed_base:
            # Floating Based Robot
            self._q[0:3] = np.copy(base_joint_pos)
            self._q[3:7] = np.copy(base_joint_quat)

            rot_w_basejoint = geom.quat_to_rot(base_joint_quat)
            twist_basejoint_in_world = np.zeros(6)
            twist_basejoint_in_world[0:3] = base_joint_ang_vel
            twist_basejoint_in_world[3:6] = base_joint_lin_vel
            augrot_joint_world = np.zeros((6, 6))
            augrot_joint_world[0:3, 0:3] = rot_w_basejoint.transpose()
            augrot_joint_world[3:6, 3:6] = rot_w_basejoint.transpose()
            twist_basejoint_in_joint = np.dot(
                augrot_joint_world, twist_basejoint_in_world
            )
            self._q_dot[0:3] = twist_basejoint_in_joint[3:6]
            self._q_dot[3:6] = twist_basejoint_in_joint[0:3]
        else:
            # Fixed Based Robot
            pass

        self._q[self.get_q_idx(list(joint_pos.keys()))] = np.copy(
            list(joint_pos.values())
        )
        self._q_dot[self.get_q_dot_idx(list(joint_vel.keys()))] = np.copy(
            list(joint_vel.values())
        )

        self._joint_positions[self.get_joint_idx(list(joint_pos.keys()))] = np.copy(
            list(joint_pos.values())
        )
        self._joint_velocities[self.get_joint_idx(list(joint_vel.keys()))] = np.copy(
            list(joint_vel.values())
        )

        self._data.qpos[:] = self._q
        self._data.qvel[:] = self._q_dot

        mujoco.mj_forward(self._model, self._data)

        if b_cent:
            self._update_centroidal_quantities()

        if self.count > 1:
            print(self.get_com_lin_vel())
            sys.exit(0)
        self.count += 1

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
        ret = np.zeros((self._model.nv, self._model.nv))
        mujoco.mj_fullM(self._model, ret, self._data.qM)
        return ret

    def get_gravity(self):
        jacp = np.zeros((3, self._model.nv), dtype=np.float64)
        mujoco.mj_jacSubtreeCom(self._model, self._data, jacp, 0)
        mass_vector = np.array([0, 0, -9.8 * mujoco.mj_getTotalmass(self._model)])
        mujoco_gravity = mass_vector @ jacp
        return mujoco_gravity

    def get_coriolis(self):
        result = np.ndarray(self.n_q_dot, dtype=float)
        mujoco.mj_rne(self._model, self._data, 0, result)
        return result - self.get_gravity()

    # CHCECKED
    def get_com_pos(self):
        mujoco.mj_comPos(self._model, self._data)
        return np.copy(self._data.subtree_com[0])

    # SOFT CHECKED
    def get_com_lin_vel(self):
        mujoco.mj_comVel(self._model, self._data)
        return np.copy(self._data.cvel[0, 3:])

    def get_com_lin_jacobian(self):
        jacp = np.zeros((3, self._model.nv), dtype=np.float64)
        mujoco.mj_jacSubtreeCom(self._model, self._data, jacp, 0)
        return jacp

    def get_com_lin_jacobian_dot(self):
        # https://github.com/google-deepmind/mujoco/issues/411
        original_qpos = self._data.qpos
        mujoco.mj_forward(self._model, self._data)
        J = self.get_com_lin_jacobian()
        h = 1e-10
        mujoco.mj_integratePos(self._model, self._data.qpos, self._data.qvel, h)
        mujoco.mj_forward(self._model, self._data)
        Jh = self.get_com_lin_jacobian()
        Jdot = (Jh - J) / h
        self._data.qpos = original_qpos
        return Jdot

    def get_link_iso(self, link_id):
        ret = np.eye(4)
        body_id = self.get_body_idx(link_id)
        ret[:3, :3] = np.reshape(self._data.xmat[body_id], (3, 3))
        ret[:3, 3] = self._data.xpos[body_id]
        return np.copy(ret)

    def get_link_vel(self, link_id):
        body_id = self.get_body_idx(link_id)
        mujoco.mj_comVel(self._model, self._data)
        return np.copy(self._data.cvel[body_id])

    def get_link_jacobian(self, link_id):
        body_id = self.get_body_idx(link_id)

        jacp = np.zeros((3, self._model.nv), dtype=np.float64)
        jacr = np.zeros((3, self._model.nv), dtype=np.float64)

        mujoco.mj_jacBody(self._model, self._data, jacp, jacr, body_id)

        ret = np.zeros((6, self._model.nv), dtype=np.float64)
        ret[0:3] = jacr
        ret[3:6] = jacp
        return np.copy(ret)

    def get_link_jacobian_dot_times_qdot(self, link_id):
        body_id = self.get_body_idx(link_id)
        jdot_qdot = self._data.cvel[body_id]

        return jdot_qdot

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
