"""Triple Pendulum Domain — swing up from hanging initial state."""

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
# Height of the torso body origin when all segments are fully upright.
# foot pivot at z=1.5, leg=0.5, thigh=0.45 => torso origin at z=2.45
_UPRIGHT_HEIGHT = 2.45

SUITE = containers.TaggedTasks()

# Hanging initial state: foot_joint = pi (leg hangs down), other joints = 0.
_INIT_QPOS = np.array([np.pi, 0.0, 0.0])
_INIT_QVEL = np.zeros(3)


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
        os.path.join(os.path.dirname(__file__), 'triple_pendulum.xml'))
    return xml, common.ASSETS


@SUITE.add('benchmarking')
def pendulum_swingup(time_limit=_DEFAULT_TIME_LIMIT,
            random=None,
            environment_kwargs=None):
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = TriplePendulum(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics for the Triple Pendulum domain."""

    def tip_height(self):
        """Z-position of the torso body origin (proxy for tip height)."""
        return self.named.data.xpos['torso', 'z']

    def torso_upright(self):
        """Cosine of angle between torso z-axis and world z-axis."""
        return self.named.data.xmat['torso', 'zz']


class TriplePendulum(base.Task):
    """Triple pendulum swingup from hanging initial state."""

    def initialize_episode(self, physics):
        physics.data.qpos[:] = _INIT_QPOS
        physics.data.qvel[:] = _INIT_QVEL
        physics.data.time = 0
        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        angles = physics.data.qpos.copy()
        obs['position'] = np.concatenate([np.sin(angles), np.cos(angles)])
        obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics):
        upright = rewards.tolerance(
            physics.tip_height(),
            bounds=(_UPRIGHT_HEIGHT * 0.9, float('inf')),
            margin=_UPRIGHT_HEIGHT)
        return upright
