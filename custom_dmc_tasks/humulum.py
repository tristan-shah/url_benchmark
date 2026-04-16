"""Humulum Domain — 2D humanoid standup from fixed initial state."""

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

_DEFAULT_TIME_LIMIT = 20
_STAND_HEIGHT = 1.2  # head height above ground considered standing

SUITE = containers.TaggedTasks()

_INIT_STATE = None


def _get_init_state():
    global _INIT_STATE
    if _INIT_STATE is None:
        npy_path = os.path.join(os.path.dirname(__file__), 'humulum_init.npy')
        _INIT_STATE = np.load(npy_path)
    return _INIT_STATE


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
        os.path.join(os.path.dirname(__file__), 'humulum.xml'))
    return xml, common.ASSETS


@SUITE.add('benchmarking')
def standup(time_limit=_DEFAULT_TIME_LIMIT,
            random=None,
            environment_kwargs=None):
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Humulum(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics for the Humulum domain."""

    def head_height(self):
        return self.named.data.xpos['head', 'z']

    def torso_upright(self):
        """Cosine of the angle between the torso z-axis and world z-axis."""
        return self.named.data.xmat['torso', 'zz']


class Humulum(base.Task):
    """2D humanoid standup from a fixed initial state."""

    def initialize_episode(self, physics):
        init = _get_init_state()
        nq = physics.model.nq
        physics.data.qpos[:] = init[:nq]
        physics.data.qvel[:] = init[nq:]
        physics.data.time = 0
        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        # Skip root_x (index 0) for translational invariance.
        obs['position'] = physics.data.qpos[1:].copy()
        obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics):
        standing = rewards.tolerance(
            physics.head_height(),
            bounds=(_STAND_HEIGHT, float('inf')),
            margin=_STAND_HEIGHT / 2)
        upright = (1 + physics.torso_upright()) / 2
        return (3 * standing + upright) / 4
