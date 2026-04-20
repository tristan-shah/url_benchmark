"""Plot stochastic head height seeds for the best skill, across one or more models."""

'''
# Single model
python plot_best_skill_seeds.py models/states/humulum/diayn/1/snapshot_2000010.pt

# Compare multiple models
python plot_best_skill_seeds.py models/states/humulum/diayn/1/snapshot_2000010.pt \
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


def find_best_skill(agent, env, num_selection_seeds=3):
    """Find the best skill by averaging over a few stochastic episodes per skill."""
    best_idx, best_mean = 0, -np.inf
    for idx in range(num_skills(agent)):
        seed_means = []
        for seed in range(num_selection_seeds):
            h = run_stochastic_episode(env, agent, idx, seed)
            seed_means.append(h.mean())
        mean = np.mean(seed_means)
        print(f'  skill {idx:>3}: mean height={mean:.3f}m')
        if mean > best_mean:
            best_mean, best_idx = mean, idx
    return best_idx


def run_stochastic_episode(env, agent, skill_idx, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    meta = make_meta(agent, skill_idx)

    time_step = env.reset()
    heights = []
    while not time_step.last():
        heights.append(env.physics.named.data.xpos['head', 'z'])
        with torch.no_grad(), utils.eval_mode(agent):
            # eval_mode=False → sample from distribution (stochastic)
            # Large step to skip uniform exploration phase
            action = agent.act(time_step.observation, meta,
                               step=agent.num_expl_steps + 1, eval_mode=False)
        time_step = env.step(action)

    return np.array(heights)


def evaluate_snapshot(snapshot, num_seeds, selection_seeds, device, env):
    payload = torch.load(snapshot, map_location=device)
    agent = payload['agent']
    agent.device = torch.device(device)

    print(f'  Finding best skill ({selection_seeds} stochastic seeds per skill)...')
    best_idx = find_best_skill(agent, env, num_selection_seeds=selection_seeds)
    print(f'  Best skill: {best_idx}')

    # Use seeds starting after the selection seeds to avoid overlap
    print(f'  Running {num_seeds} evaluation seeds for skill {best_idx}...')
    trajectories = []
    for i in range(num_seeds):
        seed = selection_seeds + i
        h = run_stochastic_episode(env, agent, best_idx, seed)
        trajectories.append(h)
        print(f'  seed {seed}: {len(h)} steps, mean={h.mean():.3f}m')

    return best_idx, np.array(trajectories)  # (num_seeds, episode_length)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshots', type=Path, nargs='+',
                        help='One or more snapshot .pt files to evaluate')
    parser.add_argument('--num-seeds', type=int, default=8,
                        help='Number of stochastic seeds for final evaluation')
    parser.add_argument('--selection-seeds', type=int, default=3,
                        help='Number of stochastic seeds used per skill during selection')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out', type=Path, default=Path('best_skill_seeds.png'))
    args = parser.parse_args()

    env = dmc.make('humulum_standup', obs_type='states',
                   frame_stack=3, action_repeat=1, seed=0)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, snapshot in enumerate(args.snapshots):
        label = model_label(snapshot)
        color = colors[i % len(colors)]
        print(f'\nEvaluating {label} ...')

        best_idx, arr = evaluate_snapshot(snapshot, args.num_seeds, args.selection_seeds, args.device, env)

        steps = np.arange(arr.shape[1])
        mean_h = arr.mean(axis=0)
        std_h = arr.std(axis=0)

        # Individual seeds (thin, faded)
        for j in range(arr.shape[0]):
            ax.plot(steps, arr[j], color=color, alpha=0.2, linewidth=0.8)

        # Mean ± std (bold)
        ax.plot(steps, mean_h, color=color, linewidth=2,
                label=f'{label} — skill {best_idx} (mean={mean_h.mean():.3f}m)')
        ax.fill_between(steps, mean_h - std_h, mean_h + std_h, color=color, alpha=0.2)

        # Save trajectories per model: (num_seeds, T)
        npy_out = args.out.parent / f'{args.out.stem}_{label.replace("/", "_")}.npy'
        np.save(npy_out, arr)
        print(f'  Saved trajectories {arr.shape} to {npy_out}')

    ax.axhline(1.2, color='k', linestyle='--', linewidth=0.8)
    title = ' vs '.join(model_label(s) for s in args.snapshots)
    ax.set_title(f'{title} — {args.num_seeds} stochastic seeds')
    ax.set_xlabel('Step')
    ax.set_ylabel('Head height (m)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f'\nSaved plot to {args.out}')


if __name__ == '__main__':
    main()
