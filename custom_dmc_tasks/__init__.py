from custom_dmc_tasks import cheetah
from custom_dmc_tasks import walker
from custom_dmc_tasks import hopper
from custom_dmc_tasks import quadruped
from custom_dmc_tasks import jaco
from custom_dmc_tasks import humulum
from custom_dmc_tasks import triple_pendulum
from custom_dmc_tasks import double_pendulum
from custom_dmc_tasks import cart_pole


def make(domain, task,
         task_kwargs=None,
         environment_kwargs=None,
         visualize_reward=False):

    if domain == 'cheetah':
        return cheetah.make(task,
                            task_kwargs=task_kwargs,
                            environment_kwargs=environment_kwargs,
                            visualize_reward=visualize_reward)
    elif domain == 'walker':
        return walker.make(task,
                           task_kwargs=task_kwargs,
                           environment_kwargs=environment_kwargs,
                           visualize_reward=visualize_reward)
    elif domain == 'hopper':
        return hopper.make(task,
                           task_kwargs=task_kwargs,
                           environment_kwargs=environment_kwargs,
                           visualize_reward=visualize_reward)
    elif domain == 'quadruped':
        return quadruped.make(task,
                           task_kwargs=task_kwargs,
                           environment_kwargs=environment_kwargs,
                           visualize_reward=visualize_reward)
    elif domain == 'humulum':
        return humulum.make(task,
                            task_kwargs=task_kwargs,
                            environment_kwargs=environment_kwargs,
                            visualize_reward=visualize_reward)
    elif domain == 'triple_pendulum':
        return triple_pendulum.make(task,
                                    task_kwargs=task_kwargs,
                                    environment_kwargs=environment_kwargs,
                                    visualize_reward=visualize_reward)
    elif domain == 'double_pendulum':
        return double_pendulum.make(task,
                                    task_kwargs=task_kwargs,
                                    environment_kwargs=environment_kwargs,
                                    visualize_reward=visualize_reward)
    elif domain == 'cart_pole':
        return cart_pole.make(task,
                              task_kwargs=task_kwargs,
                              environment_kwargs=environment_kwargs,
                              visualize_reward=visualize_reward)
    else:
        raise f'{task} not found'

    assert None
    
    
def make_jaco(task, obs_type, seed):
    return jaco.make(task, obs_type, seed)