"""Plot head height for 8 random seeds of the best skill (stochastic policy)."""

'''
python plot_best_skill_seeds.py models/states/humulum/diayn/1/snapshot_10800.pt
'''

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import dmc
import utils


def find_best_skill(agent, env, num_skills):
    """Run one deterministic episode per skill and return the best skill index."""
    best_idx, best_mean = 0, -np.inf
    for idx in range(num_skills):
        skill = np.zeros(num_skills, dtype=np.float32)
        skill[idx] = 1.0
        meta = {'skill': skill}
        time_step = env.reset()
        heights = []
        while not time_step.last():
            heights.append(env.physics.named.data.xpos['head', 'z'])
            with torch.no_grad(), utils.eval_mode(agent):
                action = agent.act(time_step.observation, meta,
                                   step=0, eval_mode=True)
            time_step = env.step(action)
        mean = np.mean(heights)
        if mean > best_mean:
            best_mean, best_idx = mean, idx
    return best_idx


def run_stochastic_episode(env, agent, skill_idx, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    skill = np.zeros(agent.skill_dim, dtype=np.float32)
    skill[skill_idx] = 1.0
    meta = {'skill': skill}

    time_step = env.reset()
    heights = []
    while not time_step.last():
        heights.append(env.physics.named.data.xpos['head', 'z'])
        with torch.no_grad(), utils.eval_mode(agent):
            # eval_mode=False → sample from distribution (stochastic)
            # Large step to skip uniform exploration, but eval_mode=False to sample stochastically
            action = agent.act(time_step.observation, meta,
                               step=agent.num_expl_steps + 1, eval_mode=False)
        time_step = env.step(action)

    return np.array(heights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--num-seeds', type=int, default=8)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out', type=Path, default=Path('best_skill_seeds.png'))
    args = parser.parse_args()

    payload = torch.load(args.snapshot, map_location=args.device)
    agent = payload['agent']
    agent.device = torch.device(args.device)

    env = dmc.make('humulum_standup', obs_type='states',
                   frame_stack=3, action_repeat=1, seed=0)

    print('Finding best skill...')
    best_idx = find_best_skill(agent, env, agent.skill_dim)
    print(f'Best skill: {best_idx}')

    print(f'Running {args.num_seeds} stochastic seeds...')
    trajectories = []
    for seed in range(args.num_seeds):
        h = run_stochastic_episode(env, agent, best_idx, seed)
        trajectories.append(h)
        print(f'  seed {seed}: {len(h)} steps, mean={h.mean():.3f}m')

    steps = np.arange(len(trajectories[0]))

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap('tab10')
    for i, h in enumerate(trajectories):
        ax.plot(np.arange(len(h)), h, color=cmap(i), alpha=0.7, label=f'seed {i}')

    # mean ± std across seeds
    arr = np.array(trajectories)  # (num_seeds, 1200)
    mean_h = arr.mean(axis=0)
    std_h = arr.std(axis=0)
    ax.plot(steps, mean_h, color='k', linewidth=2, label='mean')
    ax.fill_between(steps, mean_h - std_h, mean_h + std_h,
                    color='k', alpha=0.15, label='±1 std')

    ax.axhline(1.2, color='k', linestyle='--', linewidth=0.8)
    ax.set_title(f'{args.snapshot.name} — best skill {best_idx} ({args.num_seeds} stochastic seeds)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Head height (m)')
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    npy_out = args.out.with_suffix('.npy')
    np.save(npy_out, arr)
    print(f'Saved trajectories {arr.shape} to {npy_out}')

    fig.savefig(args.out, dpi=150)
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
