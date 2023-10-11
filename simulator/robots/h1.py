import os
import numpy as np
from .mobile import HumanoidModel

cwd = os.getcwd()

PATH_TO_ROBOT_MODEL = os.path.expanduser(cwd+'/models/robots/h1')
PATH_TO_ROBOT_XML = os.path.join(PATH_TO_ROBOT_MODEL, 'h1.xml')

class H1(HumanoidModel):
    """
    H1 is a humanoid robot developed by Unitree. 

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        super().__init__(fname=PATH_TO_ROBOT_XML, idn=idn)

    @property
    def default_gripper(self):
        """
        Since this is bimanual robot, returns dict with `'right'`, `'left'` keywords corresponding to their respective
        values

        Returns:
            dict: Dictionary containing arm-specific gripper names
        """
        return {"right": "Robotiq85Gripper", "left": "Robotiq85Gripper"}

    # @property
    # def init_qpos(self):
    #     """
    #     Since this is bimanual robot, returns [right, left] array corresponding to respective values

    #     Note that this is a pose such that the arms are half extended

    #     Returns:
    #         np.array: default initial qpos for the right, left arms
    #     """
    #     # [right, left]
    #     # Arms half extended
    #     return np.array(
    #         [0.403, -0.636, 0.114, 1.432, 0.735, 1.205, -0.269, -0.403, -0.636, -0.114, 1.432, -0.735, 1.205, 0.269]
    #     )

    @property
    def default_controller_config(self):
        return "default_panda"

    @property
    def init_qpos(self):
        return np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, np.pi / 4])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "bimanual"

    @property
    def _eef_name(self):
        """
        Since this is bimanual robot, returns dict with `'right'`, `'left'` keywords corresponding to their respective
        values

        Returns:
            dict: Dictionary containing arm-specific eef names
        """
        return {"right": "right_mount", "left": "left_mount"}
