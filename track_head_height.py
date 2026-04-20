"""Track head height over an episode for each skill, across one or more models."""

'''
# Single model
python track_head_height.py models/states/humulum/diayn/1/snapshot_2000010.pt

# Compare multiple models
python track_head_height.py models/states/humulum/diayn/1/snapshot_2000010.pt \
                            models/states/humulum/smm/1/snapshot_2000010.pt
'''

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import dmc
import utils


def model_label(snapshot: Path) -> str:
    """Derive a short label from the snapshot path, e.g. 'diayn/1'."""
    return f'{snapshot.parent.parent.name}/{snapshot.parent.name}'


def num_skills(agent) -> int:
    """Return the number of skills regardless of agent type (DIAYN vs SMM etc.)."""
    if hasattr(agent, 'skill_dim'):
        return agent.skill_dim
    if hasattr(agent, 'z_dim'):
        return agent.z_dim
    raise AttributeError(f'Cannot determine skill count from {type(agent).__name__}')


def make_meta(agent, skill_idx: int) -> dict:
    """Build the meta dict for a given skill index, handling different agent types."""
    n = num_skills(agent)
    one_hot = np.zeros(n, dtype=np.float32)
    one_hot[skill_idx] = 1.0
    if hasattr(agent, 'skill_dim'):
        return {'skill': one_hot}
    if hasattr(agent, 'z_dim'):
        return {'z': one_hot}
    raise AttributeError(f'Cannot build meta for {type(agent).__name__}')


def run_episode(env, agent, skill_idx):
    meta = make_meta(agent, skill_idx)

    time_step = env.reset()
    head_heights = []

    while not time_step.last():
        head_heights.append(env.physics.named.data.xpos['head', 'z'])
        with torch.no_grad(), utils.eval_mode(agent):
            action = agent.act(time_step.observation, meta,
                               step=0, eval_mode=True)
        time_step = env.step(action)

    return np.array(head_heights)


def evaluate_snapshot(snapshot, skill_indices, device, env):
    payload = torch.load(snapshot, map_location=device)
    agent = payload['agent']
    agent.device = torch.device(device)

    indices = skill_indices if skill_indices else list(range(num_skills(agent)))
    results = {idx: run_episode(env, agent, idx) for idx in indices}

    best_idx = max(results, key=lambda i: results[i].mean())
    best_heights = results[best_idx]

    max_len = max(len(h) for h in results.values())
    padded_arr = np.array([np.pad(h, (0, max_len - len(h)), constant_values=h[-1])
                           for h in results.values()])
    mean_heights = padded_arr.mean(axis=0)
    std_heights = padded_arr.std(axis=0)

    return results, best_idx, best_heights, mean_heights, std_heights, max_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshots', type=Path, nargs='+',
                        help='One or more snapshot .pt files to evaluate')
    parser.add_argument('--skills', type=int, nargs='+', default=None,
                        help='Skill indices to visualise (default: all)')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out', type=Path, default=Path('head_height.png'))
    args = parser.parse_args()

    env = dmc.make('humulum_standup', obs_type='states',
                   frame_stack=3, action_repeat=1, seed=0)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    title = ' vs '.join(model_label(s) for s in args.snapshots)
    fig.suptitle(title)

    for i, snapshot in enumerate(args.snapshots):
        label = model_label(snapshot)
        color = colors[i % len(colors)]
        print(f'\nEvaluating {label} ...')

        results, best_idx, best_heights, mean_heights, std_heights, max_len = \
            evaluate_snapshot(snapshot, args.skills, args.device, env)

        steps = np.arange(max_len)

        ax1.plot(best_heights, color=color,
                 label=f'{label} (skill {best_idx}, mean={best_heights.mean():.3f}m)')
        ax2.plot(steps, mean_heights, color=color, label=label)
        ax2.fill_between(steps, mean_heights - std_heights, mean_heights + std_heights,
                         color=color, alpha=0.2)

        # Save best-skill trajectory per model
        npy_out = args.out.parent / f'{args.out.stem}_{label.replace("/", "_")}.npy'
        np.save(npy_out, best_heights)
        print(f'  Saved best-skill trajectory to {npy_out}')

        print(f'  {"Skill":>6}  {"Mean height":>12}  {"Max height":>12}  {"Final height":>14}')
        for idx, heights in sorted(results.items(), key=lambda x: -x[1].mean()):
            marker = ' <-- best' if idx == best_idx else ''
            print(f'  {idx:>6}  {heights.mean():>12.3f}  {heights.max():>12.3f}  {heights[-1]:>14.3f}{marker}')

    ax1.axhline(1.2, color='k', linestyle='--', linewidth=0.8)
    ax1.set_title('Best skill per model')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Head height (m)')
    ax1.legend(fontsize=8)

    ax2.axhline(1.2, color='k', linestyle='--', linewidth=0.8)
    ax2.set_title('Average over all skills (mean ± std)')
    ax2.set_xlabel('Step')
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f'\nSaved plot to {args.out}')


if __name__ == '__main__':
    main()
