import numpy as np

from gym import utils
from gym.envs.robotics import mani_env


class ManiSlideEnv(mani_env.ManiEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.0,
            'robot0:slide1': 0.0,
            'robot0:slide2': 0.0,
            'robot0:joint1': np.pi / 2,
            'robot0:joint2': np.pi / 6,
            'robot0:joint4': np.pi / 3,
            'robot0:joint6': np.pi / 2,
            'object0:joint': [0.0, 5.3, 0.6, 1., 0., 0., 0.],
        }
        mani_env.ManiEnv.__init__(
            self, 'mani/mani_slide.xml', has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=-0.02, target_in_the_air=False, target_offset=np.array([0.0, 11.0, 0.0]),
            obj_range=1, target_range=2.0, distance_threshold=0.5, holding_block=False, fixed_range=False,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
