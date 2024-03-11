import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np
from util import geom
from simulator.envs import DoorEnv, EmptyEnv
from simulator.render import CV2Renderer
from simulator.recorder import HDF5Recorder
import time
import argparse
import mujoco
import hid
import threading
# from mujoco import viewer
import cv2

import ipdb

class SpaceMouse():
    """
    A minimalistic driver class for SpaceMouse with HID library.
    Note: Use hid.enumerate() to view all USB human interface devices (HID).
    Make sure SpaceMouse is detected before running the script.
    You can look up its vendor/product id from this method.
    Args:
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of scale input rotation commands scaling
    """

    def __init__(self,
                 pos_sensitivity=6.0,
                 rot_sensitivity=6.0
                 ):

        print("Opening SpaceMouse device")
        found = False
        for device_info in hid.enumerate():
            print(device_info)
            if device_info['product_string'] == 'SpaceMouse Wireless':
                self.info = device_info
                found = True
                break
        if found:
            print("Found SpaceMouse")
            print("Manufacturer: %s" % self.info['vendor_id'])
            print("Product: %s" % self.info['product_id'])
        else:
            print("Could not find SpaceMouse")
            return

        print("Opening SpaceMouse device")
        self.device = hid.Device(self.info['vendor_id'], self.info['product_id'])

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        # 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self.single_click_and_hold = False

        self._control = np.zeros(6)
        self._reset_state = 0
        self._enabled = False

        # launch a new listener thread to listen to SpaceMouse
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()


    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        # Reset 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0
        # Reset control
        self._control = np.zeros(6)
        # Reset grasp
        self.single_click_and_hold = False


    def run(self):
        """Listener method that keeps pulling new messages."""

        t_last_click = -1

        while True:
            d = self.device.read(13)

            if d is not None:

                if d[0] == 1:  ## readings from 6-DoF sensor
                    self.y = int.from_bytes(d[1:3], "little", signed=True)
                    self.x = int.from_bytes(d[3:5], "little", signed=True)
                    self.z = int.from_bytes(d[5:7], "little", signed=True)

                    self.roll = int.from_bytes(d[7:9], "little", signed=True)
                    self.pitch = int.from_bytes(d[9:11], "little", signed=True)
                    self.yaw = int.from_bytes(d[11:13], "little", signed=True)

                    self._control = - np.array([self.x,
                                                self.y,
                                                self.z,
                                                self.roll,
                                                self.pitch,
                                                self.yaw])/350.

                elif d[0] == 3:  ## readings from the side buttons

                    # press left button
                    if d[1] == 1:
                        t_click = time.time()
                        elapsed_time = t_click - t_last_click
                        t_last_click = t_click
                        self.single_click_and_hold = True

                    # release left button
                    if d[1] == 0:
                        self.single_click_and_hold = False

                    # right button is for reset
                    if d[1] == 2:
                        self._reset_state = 1
                        self._enabled = False
                        self._reset_internal_state()

    @property
    def control(self):
        """
        Grabs current pose of Spacemouse
        Returns:
            np.array: 6-DoF control value
        """
        return self._control

    @property
    def click(self):
        """
        Maps internal states into gripper commands.
        Returns:
            float: Whether we're using single click and hold or not
        """
        if self.single_click_and_hold:
            return 1.0
        return 0

RIGHTFORWARD_GRIPPER = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])
TRANSFORM_VR = np.array(
    [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
)  # geom.euler_to_rot(np.array([0, np.pi, 0]))

ENV_LOOKUP = {
    "door": DoorEnv,
}

def main(gui, env_type, cam_name="upview", subtask=0, save_video=True):
    env_class = EmptyEnv

    env = env_class()

    env.config["Manipulation"]["Trajectory Mode"] = "interpolation"

    if save_video:
        save_path = os.path.join(
            ".", "{}_{}_{}.mp4".format(env_type, cam_name, subtask)
        )
    else:
        save_path = None
    renderer = CV2Renderer(
        device_id=-1, sim=env.sim, cam_name=cam_name, gui=gui, save_path=save_path,
        width=1600,
        height=1600,
    )
    # recorder = None
    recorder = HDF5Recorder(
        sim=env.sim,
        config=env.config,
        file_path="./test/demo_{}_{}".format(env_type, int(time.time())),
    )

    env.set_renderer(renderer)
    env.set_recorder(recorder)
    env.reset(subtask=subtask)

    done = False
    subtask = 0
    left_pos =  np.array([.1, 0.1 , .1])
    right_default_pos =  np.array([.1, -0.1 , .1])
    init_time = env.cur_time
    space_mouse = SpaceMouse()

    while not done:

        action = {}
        action['trajectory'] = {}
        action['gripper'] = {}
        action['aux'] = {}
        action['subtask'] = 0
        action['locomotion'] = 0

        lh_target_pos = left_pos
        lh_input = geom.euler_to_rot(np.array([0, 0, 0]))
        lh_target_rot = np.dot(lh_input, RIGHTFORWARD_GRIPPER)

        inp = space_mouse.control
        rh_target_pos = np.array([inp[0], inp[1], inp[2]]) + right_default_pos
        rh_rotation_input = np.array([inp[3], inp[4], inp[5]])
        rh_target_rot = geom.euler_to_rot(np.dot(rh_rotation_input, RIGHTFORWARD_GRIPPER))

        action['trajectory']['left_pos'] = lh_target_pos
        action['trajectory']['right_pos'] = rh_target_pos
        action['trajectory']['right_quat'] = geom.rot_to_quat(rh_target_rot)
        action['trajectory']['left_quat'] = geom.rot_to_quat(lh_target_rot)
        
        # frequency of actions? 
        env.step(action)

        if env.cur_time > 20.0:
            done = True

    recorder.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", type=int, default=1, help="")
    parser.add_argument("--env", type=str, default="door", help="")
    parser.add_argument("--cam", type=str, default="upview", help="")
    parser.add_argument("--subtask", type=int, default=0, help="")
    args = parser.parse_args()

    gui = args.gui
    env_type = args.env
    cam_name = args.cam
    subtask = args.subtask

    main(gui=gui, env_type=env_type, cam_name=cam_name, subtask=subtask)
