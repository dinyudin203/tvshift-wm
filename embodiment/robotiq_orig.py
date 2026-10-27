"""Upstream robosuite v1.4.1 Robotiq 2F-85 (robotiq_gripper_85.xml: rotating knuckles coupled by tendons), copied verbatim
from ARISE-Initiative/robosuite@v1.4.1 models/grippers/robotiq_85_gripper.py and registered as "Robotiq85GripperOrig".
The installed robosuite was patched to load a slide-joint Menagerie variant under the "Robotiq85Gripper" name."""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Robotiq85GripperOrigBase(GripperModel):
    """
    6-DoF Robotiq gripper.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/robotiq_gripper_85.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([-0.026, -0.267, -0.200, -0.026, -0.267, -0.200])

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "left_outer_finger_collision",
                "left_inner_finger_collision",
                "left_fingertip_collision",
                "left_fingerpad_collision",
            ],
            "right_finger": [
                "right_outer_finger_collision",
                "right_inner_finger_collision",
                "right_fingertip_collision",
                "right_fingerpad_collision",
            ],
            "left_fingerpad": ["left_fingerpad_collision"],
            "right_fingerpad": ["right_fingerpad_collision"],
        }


class Robotiq85GripperOrig(Robotiq85GripperOrigBase):
    """
    1-DoF variant of RobotiqGripperBase.
    """

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == 1
        self.current_action = np.clip(self.current_action + self.speed * np.sign(action), -1.0, 1.0)
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1


from robosuite.models.grippers import GRIPPER_MAPPING
GRIPPER_MAPPING["Robotiq85GripperOrig"] = Robotiq85GripperOrig


class Robotiq85GripperMenagerie(GripperModel):
    """MuJoCo Menagerie Robotiq 2F-85 with its real 4-bar linkage (grippers/robotiq_2f85_menagerie_orig), 1-DoF
    open/close with the same action handling as robosuite's Robotiq85Gripper."""

    def __init__(self, idn=0):
        import os
        super().__init__(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "grippers/robotiq_2f85_menagerie_orig/2f85_menagerie_robosuite.xml"), idn=idn)

    def format_action(self, action):
        assert len(action) == 1
        self.current_action = np.clip(self.current_action + self.speed * np.sign(action), -1.0, 1.0)
        return self.current_action

    @property
    def init_qpos(self):
        return np.zeros(8)

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


GRIPPER_MAPPING["Robotiq85GripperMenagerie"] = Robotiq85GripperMenagerie


class Robotiq85GripperMenagerieLocalFinger(Robotiq85GripperMenagerie):
    """Menagerie 4-bar 2F-85 with the finger values of robosuite's 2f85_robosuite.xml (pad friction 0.8/0.7, pad
    inertia 0.02 kg, actuator forcerange +-20)."""

    def __init__(self, idn=0):
        import os
        GripperModel.__init__(self, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "grippers/robotiq_2f85_menagerie_orig/2f85_menagerie_localfinger_robosuite.xml"), idn=idn)


GRIPPER_MAPPING["Robotiq85GripperMenagerieLocalFinger"] = Robotiq85GripperMenagerieLocalFinger


class Robotiq85GripperInnerSlide(GripperModel):
    """robosuite's 2f85_robosuite.xml (slide pads) with the outer finger fixed to the base and only the inner finger
    (follower + pads) sliding closed (grippers/robotiq_2f85_inner_slide). Same physics and action handling as the
    installed Robotiq85Gripper."""

    def __init__(self, idn=0):
        import os
        super().__init__(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "grippers/robotiq_2f85_inner_slide/2f85_inner_slide_robosuite.xml"), idn=idn)

    def format_action(self, action):
        assert len(action) == 1
        self.current_action = np.clip(self.current_action + self.speed * np.sign(action), -1.0, 1.0)
        return self.current_action

    @property
    def init_qpos(self):
        return np.zeros(8)

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


GRIPPER_MAPPING["Robotiq85GripperInnerSlide"] = Robotiq85GripperInnerSlide


class Robotiq85GripperMenagerieNoLinkCol(Robotiq85GripperMenagerie):
    """Robotiq85GripperMenagerieLocalFinger with the linkage collision meshes disabled (only pads and base collide)."""

    def __init__(self, idn=0):
        import os
        GripperModel.__init__(self, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "grippers/robotiq_2f85_menagerie_orig/2f85_menagerie_localfinger_nolinkcol_robosuite.xml"), idn=idn)


GRIPPER_MAPPING["Robotiq85GripperMenagerieNoLinkCol"] = Robotiq85GripperMenagerieNoLinkCol
