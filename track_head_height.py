"""Track head height over an episode for each DIAYN skill."""

'''
python track_head_height.py models/states/humulum/diayn/1/snapshot_10800.pt
'''

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import dmc
import utils


def run_episode(env, agent, skill_idx):
    skill = np.zeros(agent.skill_dim, dtype=np.float32)
    skill[skill_idx] = 1.0
    meta = {'skill': skill}

    time_step = env.reset()
    head_heights = []

    while not time_step.last():
        head_heights.append(env.physics.named.data.xpos['head', 'z'])
        with torch.no_grad(), utils.eval_mode(agent):
            action = agent.act(time_step.observation, meta,
                               step=0, eval_mode=True)
        time_step = env.step(action)

    return np.array(head_heights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot', type=Path,
                        help='Path to snapshot .pt file')
    parser.add_argument('--skills', type=int, nargs='+', default=None,
                        help='Skill indices to visualise (default: all)')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out', type=Path, default=Path('head_height.png'))
    args = parser.parse_args()

    payload = torch.load(args.snapshot, map_location=args.device)
    agent = payload['agent']
    agent.device = torch.device(args.device)

    env = dmc.make('humulum_standup', obs_type='states',
                   frame_stack=3, action_repeat=1, seed=0)

    skill_indices = args.skills if args.skills else list(range(agent.skill_dim))

    results = {idx: run_episode(env, agent, idx) for idx in skill_indices}

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, heights in results.items():
        ax.plot(heights, label=f'skill {idx}', alpha=0.75)

    ax.axhline(1.2, color='k', linestyle='--', linewidth=0.8, label='stand threshold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Head height (m)')
    ax.set_title(f'Head height per skill — {args.snapshot.name}')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f'Saved to {args.out}')

    print(f'\n{"Skill":>6}  {"Max height":>12}  {"Final height":>14}')
    for idx, heights in results.items():
        print(f'{idx:>6}  {heights.max():>12.3f}  {heights[-1]:>14.3f}')


if __name__ == '__main__':
    main()
