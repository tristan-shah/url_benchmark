"""Evaluate extremity height for skill-discovery agents across environments.

Finds the best skill (highest mean extremity height), then runs stochastic
rollouts with that skill and saves a (n_seeds, T) array per snapshot.

Usage
-----
# Single snapshot — domain inferred from path
python eval_extremity_height.py models/states/humulum/diayn/1/snapshot_2000010.pt

# Multiple snapshots compared side-by-side
python eval_extremity_height.py \\
    models/states/humulum/diayn/1/snapshot_2000010.pt \\
    models/states/humulum/smm/1/snapshot_2000010.pt

# Override domain explicitly
python eval_extremity_height.py ... --domain triple_pendulum

# Saved .npy shape: (n_seeds, T) — compatible with the external plotting framework
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import dmc
import utils


# ---------------------------------------------------------------------------
# Environment registry
# Add new environments here; nothing else needs to change.
#
# Each entry:
#   task        : name passed to dmc.make()
#   height_fn   : callable(env) -> float  — current extremity height in metres
#   ref_height  : float | None — drawn as a dashed reference line on the plot
# ---------------------------------------------------------------------------
REGISTRY = {
    'humulum': {
        'task': 'humulum_standup',
        'height_fn': lambda env: float(env.physics.named.data.xpos['head', 'z']),
        'ref_height': 1.2,
    },
    'triple_pendulum': {
        'task': 'triple_pendulum_swingup',
        # tip of the torso geom: body-origin z + 0.4 * cos(angle from world-z)
        'height_fn': lambda env: float(
            env.physics.named.data.xpos['torso', 'z'] +
            0.4 * env.physics.named.data.xmat['torso', 'zz']
        ),
        'ref_height': None,
    },
    'double_pendulum': {
        'task': 'double_pendulum_swingup',
        # tip of second_link: body-origin z + R @ [0,0,-1] z-component
        'height_fn': lambda env: float(
            env.physics.named.data.xpos['second_link', 'z'] -
            env.physics.named.data.xmat['second_link', 'zz']
        ),
        'ref_height': None,
    },
    'cart_pole': {
        'task': 'cart_pole_swingup',
        # tip of pole geom: body-origin z + R @ [0,0,-1] z-component
        'height_fn': lambda env: float(
            env.physics.named.data.xpos['pole', 'z'] -
            env.physics.named.data.xmat['pole', 'zz']
        ),
        'ref_height': None,
    },
}


# ---------------------------------------------------------------------------
# Agent helpers (agent-type-agnostic)
# ---------------------------------------------------------------------------

def is_skill_agent(agent) -> bool:
    return hasattr(agent, 'skill_dim') or hasattr(agent, 'z_dim')


def num_skills(agent) -> int:
    if hasattr(agent, 'skill_dim'):
        return agent.skill_dim
    if hasattr(agent, 'z_dim'):
        return agent.z_dim
    raise AttributeError(f'Cannot determine skill count from {type(agent).__name__}')


def make_meta(agent, skill_idx: int) -> dict:
    n = num_skills(agent)
    one_hot = np.zeros(n, dtype=np.float32)
    one_hot[skill_idx] = 1.0
    if hasattr(agent, 'skill_dim'):
        return {'skill': one_hot}
    if hasattr(agent, 'z_dim'):
        return {'z': one_hot}
    raise AttributeError(f'Cannot build meta for {type(agent).__name__}')


def model_label(snapshot: Path) -> str:
    return f'{snapshot.parent.parent.name}/{snapshot.parent.name}'


def domain_from_path(snapshot: Path) -> str:
    """Infer domain from path: models/states/{domain}/{agent}/{seed}/snapshot.pt"""
    return snapshot.parent.parent.parent.name


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, agent, height_fn, seed: int,
                meta: dict = None) -> np.ndarray:
    np.random.seed(seed)
    torch.manual_seed(seed)

    if meta is None:
        meta = agent.init_meta()
    time_step = env.reset()
    heights = []

    while not time_step.last():
        heights.append(height_fn(env))
        with torch.no_grad(), utils.eval_mode(agent):
            action = agent.act(
                time_step.observation, meta,
                step=agent.num_expl_steps + 1,
                eval_mode=False,
            )
        time_step = env.step(action)

    return np.array(heights, dtype=np.float32)


# ---------------------------------------------------------------------------
# Skill selection + evaluation
# ---------------------------------------------------------------------------

def find_best_skill(agent, env, height_fn, selection_seeds: int) -> int:
    best_idx, best_mean = 0, -np.inf
    for idx in range(num_skills(agent)):
        meta = make_meta(agent, idx)
        means = [run_episode(env, agent, height_fn, seed=s, meta=meta).mean()
                 for s in range(selection_seeds)]
        mean = float(np.mean(means))
        print(f'    skill {idx:>3}: mean height = {mean:.4f} m')
        if mean > best_mean:
            best_mean, best_idx = mean, idx
    return best_idx


def evaluate_snapshot(snapshot: Path, height_fn, num_seeds: int,
                      selection_seeds: int, device: str, env):
    payload = torch.load(snapshot, map_location=device)
    agent = payload['agent']
    agent.device = torch.device(device)

    if is_skill_agent(agent):
        print(f'  Finding best skill ({selection_seeds} seeds per skill) ...')
        best_idx = find_best_skill(agent, env, height_fn, selection_seeds)
        print(f'  Best skill: {best_idx}')
        meta = make_meta(agent, best_idx)
        skill_label = f'skill {best_idx}'
        seed_offset = selection_seeds
    else:
        print(f'  No skills (non-skill agent) — running {num_seeds} seeds directly.')
        best_idx = None
        meta = agent.init_meta()
        skill_label = 'no skill'
        seed_offset = 0

    print(f'  Running {num_seeds} evaluation seeds ({skill_label}) ...')
    trajs = []
    for i in range(num_seeds):
        h = run_episode(env, agent, height_fn, seed=seed_offset + i, meta=meta)
        trajs.append(h)
        print(f'    seed {seed_offset + i}: {len(h)} steps, mean = {h.mean():.4f} m')

    max_len = max(len(t) for t in trajs)
    arr = np.array([np.pad(t, (0, max_len - len(t)), constant_values=t[-1])
                    for t in trajs], dtype=np.float32)  # (n_seeds, T)

    return best_idx, skill_label, arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate extremity height for skill-discovery agents.')
    parser.add_argument('snapshots', type=Path, nargs='+')
    parser.add_argument('--domain', default=None,
                        help='Override domain (default: inferred from snapshot path)')
    parser.add_argument('--num-seeds', type=int, default=10,
                        help='Stochastic evaluation seeds for best skill (default: 10)')
    parser.add_argument('--selection-seeds', type=int, default=3,
                        help='Seeds used per skill during best-skill selection (default: 3)')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out', type=Path, default=Path('extremity_height.png'))
    args = parser.parse_args()

    # Resolve domain and environment config
    domain = args.domain or domain_from_path(args.snapshots[0])
    if domain not in REGISTRY:
        raise ValueError(
            f'Unknown domain "{domain}". '
            f'Available: {list(REGISTRY.keys())}. '
            f'Add it to REGISTRY in this file.'
        )
    cfg = REGISTRY[domain]
    height_fn = cfg['height_fn']

    print(f'Domain: {domain}  |  Task: {cfg["task"]}')
    env = dmc.make(cfg['task'], obs_type='states', frame_stack=1,
                   action_repeat=1, seed=0)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, (ax_best, ax_mean) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(f'{domain}  —  {" vs ".join(model_label(s) for s in args.snapshots)}')

    for i, snapshot in enumerate(args.snapshots):
        label = model_label(snapshot)
        color = colors[i % len(colors)]
        print(f'\nEvaluating {label} ...')

        best_idx, skill_label, arr = evaluate_snapshot(
            snapshot, height_fn, args.num_seeds, args.selection_seeds,
            args.device, env,
        )

        steps = np.arange(arr.shape[1])
        mean_h = arr.mean(axis=0)
        std_h = arr.std(axis=0)

        # Best-skill panel: individual seeds (faded) + mean (bold)
        for j in range(arr.shape[0]):
            ax_best.plot(steps, arr[j], color=color, alpha=0.2, linewidth=0.7)
        ax_best.plot(steps, mean_h, color=color, linewidth=2,
                     label=f'{label} — {skill_label} (mean={mean_h.mean():.4f} m)')

        # Mean ± std panel
        ax_mean.plot(steps, mean_h, color=color, label=label)
        ax_mean.fill_between(steps, mean_h - std_h, mean_h + std_h,
                             color=color, alpha=0.2)

        # Save (n_seeds, T) array — matches external plotting framework format
        stem = args.out.stem
        safe_label = label.replace('/', '_')
        npy_out = args.out.parent / f'{stem}_{domain}_{safe_label}.npy'
        np.save(npy_out, arr)
        print(f'  Saved trajectories {arr.shape} -> {npy_out}')

        print(f'\n  Summary for {label}:')
        print(f'    {"Metric":<20} {"Value":>10}')
        print(f'    {"Mean height (m)":<20} {mean_h.mean():>10.4f}')
        print(f'    {"Max height (m)":<20} {arr.max():>10.4f}')
        print(f'    {"Final height (m)":<20} {mean_h[-1]:>10.4f}')

    for ax in (ax_best, ax_mean):
        if cfg['ref_height'] is not None:
            ax.axhline(cfg['ref_height'], color='k', linestyle='--',
                       linewidth=0.8, label=f'ref ({cfg["ref_height"]} m)')

    ax_best.set_title(f'Best skill — {args.num_seeds} stochastic seeds')
    ax_best.set_xlabel('Step')
    ax_best.set_ylabel('Extremity height (m)')
    ax_best.legend(fontsize=8)

    ax_mean.set_title('Mean ± std across seeds')
    ax_mean.set_xlabel('Step')
    ax_mean.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f'\nSaved plot -> {args.out}')


if __name__ == '__main__':
    main()
