DOMAINS = [
    'walker',
    'quadruped',
    'jaco',
    'humulum',
    'triple_pendulum',
    'double_pendulum',
]

WALKER_TASKS = [
    'walker_stand',
    'walker_walk',
    'walker_run',
    'walker_flip',
]

QUADRUPED_TASKS = [
    'quadruped_walk',
    'quadruped_run',
    'quadruped_stand',
    'quadruped_jump',
]

JACO_TASKS = [
    'jaco_reach_top_left',
    'jaco_reach_top_right',
    'jaco_reach_bottom_left',
    'jaco_reach_bottom_right',
]

HUMULUM_TASKS = [
    'humulum_standup',
]

TRIPLE_PENDULUM_TASKS = [
    'triple_pendulum_swingup',
]

DOUBLE_PENDULUM_TASKS = [
    'double_pendulum_swingup',
]

TASKS = WALKER_TASKS + QUADRUPED_TASKS + JACO_TASKS + HUMULUM_TASKS + TRIPLE_PENDULUM_TASKS + DOUBLE_PENDULUM_TASKS

PRIMAL_TASKS = {
    'walker': 'walker_stand',
    'jaco': 'jaco_reach_top_left',
    'quadruped': 'quadruped_walk',
    'humulum': 'humulum_standup',
    'triple_pendulum': 'triple_pendulum_swingup',
    'double_pendulum': 'double_pendulum_swingup',
}
