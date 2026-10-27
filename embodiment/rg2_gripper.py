"""OnRobot RG2 (grippers/onrobot_rg2/rg2_robosuite.xml: parallel 4-bar via joint equalities), registered as "OnRobotRG2Gripper".
Same 1-DoF action handling as the Robotiq85GripperMenagerie* grippers in robotiq_orig.py (-1 open, +1 closed, slew by speed)."""
import os
import numpy as np

from robosuite.models.grippers import GRIPPER_MAPPING
from robosuite.models.grippers.gripper_model import GripperModel


class OnRobotRG2Gripper(GripperModel):
    def __init__(self, idn=0):
        super().__init__(os.path.join(os.path.dirname(os.path.abspath(__file__)), "grippers/onrobot_rg2/rg2_robosuite.xml"), idn=idn)

    def format_action(self, action):
        assert len(action) == 1
        self.current_action = np.clip(self.current_action + self.speed * np.sign(action), -1.0, 1.0)
        return self.current_action

    @property
    def init_qpos(self):
        # joint order: left driver, left inner finger, left inner knuckle, right driver, right inner finger, right inner knuckle (open)
        return np.array([-0.1, 0.1, -0.1, -0.1, 0.1, -0.1])

    @property
    def speed(self):
        return 0.20

    @property
    def dof(self):
        return 1

    @property
    def _important_geoms(self):
        return {"left_finger": ["left_pad1", "left_pad2"], "right_finger": ["right_pad1", "right_pad2"],
                "left_fingerpad": ["left_pad1", "left_pad2"], "right_fingerpad": ["right_pad1", "right_pad2"]}


GRIPPER_MAPPING["OnRobotRG2Gripper"] = OnRobotRG2Gripper
