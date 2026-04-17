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

    best_idx = max(results, key=lambda i: results[i].mean())
    best_heights = results[best_idx]

    # pad to common length then average
    max_len = max(len(h) for h in results.values())
    padded = [np.pad(h, (0, max_len - len(h)), constant_values=h[-1])
              for h in results.values()]
    padded_arr = np.array(padded)
    mean_heights = np.mean(padded_arr, axis=0)
    std_heights = np.std(padded_arr, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(args.snapshot.name)

    ax1.plot(best_heights, color='tab:blue')
    ax1.axhline(1.2, color='k', linestyle='--', linewidth=0.8)
    ax1.set_title(f'Best skill (skill {best_idx}, mean={best_heights.mean():.3f}m)')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Head height (m)')

    steps = np.arange(max_len)
    ax2.plot(steps, mean_heights, color='tab:orange', label='mean')
    ax2.fill_between(steps, mean_heights - std_heights, mean_heights + std_heights,
                     color='tab:orange', alpha=0.3, label='±1 std')
    ax2.axhline(1.2, color='k', linestyle='--', linewidth=0.8)
    ax2.set_title(f'Average over all {len(skill_indices)} skills')
    ax2.set_xlabel('Step')
    ax2.legend()

    npy_out = args.out.with_suffix('.npy')
    np.save(npy_out, best_heights)
    print(f'Saved trajectory to {npy_out}')

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f'Saved to {args.out}')

    print(f'\n{"Skill":>6}  {"Mean height":>12}  {"Max height":>12}  {"Final height":>14}')
    for idx, heights in sorted(results.items(), key=lambda x: -x[1].mean()):
        marker = ' <-- best' if idx == best_idx else ''
        print(f'{idx:>6}  {heights.mean():>12.3f}  {heights.max():>12.3f}  {heights[-1]:>14.3f}{marker}')


if __name__ == '__main__':
    main()
