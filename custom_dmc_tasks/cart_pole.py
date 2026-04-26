"""Cart-Pole Domain — swing up from hanging initial state."""

import collections
import os

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import io as resources

_DEFAULT_TIME_LIMIT = 12
# Pole tip z when fully upright: cart z=1.1 + pole offset z=0.05 + pole length 1.0 = 2.15 m
_UPRIGHT_TIP_HEIGHT = 2.15

SUITE = containers.TaggedTasks()

# Hanging initial state: cart centred, pole hanging down (hinge=0).
# fromto="0 0 0 0 0 -1" means pole points down at qpos=0.
_INIT_QPOS = np.zeros(2)   # [cart_slide=0, pole_hinge=0]
_INIT_QVEL = np.zeros(2)


def make(task,
         task_kwargs=None,
         environment_kwargs=None,
         visualize_reward=False):
    task_kwargs = task_kwargs or {}
    if environment_kwargs is not None:
        task_kwargs = task_kwargs.copy()
        task_kwargs['environment_kwargs'] = environment_kwargs
    env = SUITE[task](**task_kwargs)
    env.task.visualize_reward = visualize_reward
    return env


def get_model_and_assets():
    xml = resources.GetResource(
        os.path.join(os.path.dirname(__file__), 'cart_pole.xml'))
    return xml, common.ASSETS


@SUITE.add('benchmarking')
def pole_swingup(time_limit=_DEFAULT_TIME_LIMIT,
                 random=None,
                 environment_kwargs=None):
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = CartPole(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics for the Cart-Pole domain."""

    def pole_tip_height(self):
        """Z-position of the pole tip (end of pole geom in world frame)."""
        return (self.named.data.xpos['pole', 'z'] -
                self.named.data.xmat['pole', 'zz'])

    def cart_position(self):
        return self.named.data.qpos['cart_slide']


class CartPole(base.Task):
    """Cart-pole swingup from hanging initial state."""

    def initialize_episode(self, physics):
        physics.data.qpos[:] = _INIT_QPOS
        physics.data.qvel[:] = _INIT_QVEL
        physics.data.time = 0
        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        pole_angle = physics.data.qpos[1]
        # cart_slide is translational — use raw value (bounded by stiffness spring)
        # pole_hinge is rotational (unlimited) — use sin/cos
        obs['position'] = np.array([
            physics.data.qpos[0],    # cart x-position
            np.sin(pole_angle),
            np.cos(pole_angle),
        ], dtype=np.float32)
        obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics):
        return rewards.tolerance(
            physics.pole_tip_height(),
            bounds=(_UPRIGHT_TIP_HEIGHT * 0.9, float('inf')),
            margin=_UPRIGHT_TIP_HEIGHT)
