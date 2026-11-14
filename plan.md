# Plan

Deterministic Policy Gradient Algorithms — E1 277 Reinforcement Learning.
How the work in `spec.md` will be built. `task.md` is the checklist that carries this plan out. `code.md` is the formula-level contract.

Source of the dates and the team split: `RL_E1277o/CourseProject/ProjectPlan/DPG_Project_Plan.md`, 24 Oct 2026.
The Pendulum experiment was removed from this plan on 14 Nov 2026. The remaining environments are the quadratic bandit and LunarLanderContinuous-v3.

## What this project is arguing

Each algorithm exists to repair the one before it. The code and the plots have to make that chain visible.

1. Stochastic Actor-Critic (Gaussian) is the 2014 baseline. It estimates an integral over actions. Call it by that name. Never call it SAC.
2. COPDAC-Q deletes that integral. The actor steps along the slope of \(Q\) at the deterministic action. On the bandit this should win by more as the action dimension goes from 10 to 25 to 50.
3. DDPG keeps that same gradient and replaces the linear critic with networks, a replay buffer, and target networks, and is run on LunarLander.
4. TD3 keeps DDPG and stops the critic from overestimating, with twin critics, a delayed actor, and smoothed targets.

Natural-gradient COPDAC is a cheap control, not new research. It is the paper's own one-line actor change, \(\theta \leftarrow \theta + \alpha w\). On the bandit that step matches the ordinary update, and the runs still get their own result folder. OffPAC is not scheduled.

## How the code is split

One process, one interface, three owners. Shared types are agreed before anyone writes an update rule.

```
run.py                the only training entry point
envs/bandit.py        A: quadratic cost, dimensions 10 / 25 / 50
agents/stochastic_ac  A
agents/offpac         unused stub
agents/copdac_q       B, natural gradient is a flag
agents/ddpg_td3       C, TD3 is three flags on the same class
configs/*.yaml        one file per agent and environment that is actually run
results/              one CSV and a copy of the resolved config per seed
plots/make_plots.py   the only way numbers reach a figure or a table
```

A and B share the bandit, so their comparison is stochastic against deterministic on the same cost. C never reimplements the linear agents. B never reimplements the networks.

An add-on is started only after that owner's main sanity check passes. If someone falls behind, their add-on is dropped first. TD3 is the last add-on to drop, because it is the third step of the story.

## Build order

The phases match `task.md`. Dates are the plan's, not a reason to skip a failed check.

| Phase | When | What is true at the end |
| --- | --- | --- |
| 0 Setup | 25–26 Oct | The environment installs, the bandit cost is correct, LunarLander steps, the plot script can draw a dummy band. |
| 1 Sanity | 27–29 Oct | Stochastic Actor-Critic and COPDAC-Q learn the \(m = 10\) bandit. |
| 2 Matrix and add-ons | 30 Oct–1 Nov | Bandit curves at 10, 25, and 50 for five seeds. Natural-gradient COPDAC on that grid. |
| 3 LunarLander | 1–4 Nov | LunarLander is DDPG against TD3 only. |
| Freeze | 5 Nov | Plots and the results table come from one script. Configs sit next to CSVs. |
| Writing | 6–15 Nov | Not a coding phase. Drafts 10 Nov, report and slides 12 Nov, rehearsal 13 Nov, presentation 15 Nov. |

## Experiment plan

Five seeds, 0 through 4, for every final cell. Mean and standard deviation across seeds. Evaluate the mean or deterministic policy, with exploration off, every 5,000 steps. On the bandit, evaluate once per batch of \(2m\) steps, and record the cost of the mean action.

| Environment | Who runs | Steps | Question |
| --- | --- | --- | --- |
| Bandit, \(m = 10, 25, 50\) | Stochastic Actor-Critic, COPDAC-Q, natural-gradient COPDAC. DDPG is optional and not required. | 10,000 | Does the deterministic gradient win by more as dimension grows? |
| LunarLanderContinuous-v3 | DDPG and TD3 only. | about 300,000 | Does TD3 fix DDPG's instability? |

Linear step sizes are swept, 3 or 4 values, on seed 0 only, before the five-seed run. DDPG and TD3 share one hyperparameter table. The only legal difference between those two runs is TD3's three flags.

Metrics the logs must be able to support:

- Learning curve against environment steps.
- Steps until LunarLander return exceeds 200. If a run never crosses, the table says so.
- Wall-clock seconds.
- Seed spread, which is the DDPG against TD3 comparison.
- Optional: variance of 100 gradient estimates at a fixed bandit parameter, and the critic's predicted \(Q\) against the realised return.

## Decisions already made

These are not open during implementation.

- Bandit critic for both linear methods: least squares every \(2m\) steps, one actor step per batch. The regression target is the reward, which is the negative cost.
- COPDAC-GQ is the cited repair if the linear critic diverges. It is not code, unless `spec.md` is edited.
- Exploration noise for the neural agents is Gaussian, added after the tanh squash. No Ornstein–Uhlenbeck process.
- Time-limit cuts are truncations. The update still bootstraps. Only a true terminal state sets `done`.
- At \(m = 10\) seed 0, COPDAC-Q's final cost (0.0011) is higher than Stochastic Actor-Critic's (2.1e-5). That result was accepted on 14 Nov. The dimension argument is the five-seed grid, where COPDAC-Q wins at \(m = 25\) and \(m = 50\).
- Results files are written by the runner and read by the plot script. Nobody types a number into a CSV or a slide table.

## Fallbacks

Use the first row that matches. Record the choice in the config and in the task note.

| Early sign | Fallback |
| --- | --- |
| COPDAC-Q weights or \(Q\) become NaN | Lower the critic step size by 3 to 10 times. |
| One LunarLander run exceeds 1.5 hours | Run seeds in parallel, or cut that config to 200,000 steps. |
| Box2D will not import | Install `swig`, then the Gymnasium extra. Colab is allowed for these seeds. Do not swap the environment. |
| TD3 does not beat DDPG on LunarLander | Report the overlap, the seed band, and the Q-overestimation plot if it exists. |

## What the presentation needs from the code

Ten slides, fifteen minutes, then questions. One main plot per results slide. Extra plots are backup slides and do not count toward the ten.

The plots that must exist by the freeze, if the matching runs were kept:

- Bandit learning curves at three action dimensions, Stochastic Actor-Critic against COPDAC-Q, with the natural-gradient runs if they finished.
- LunarLander, DDPG against TD3, with the seed band.

The report, not the code, holds the derivations. Theory slides carry at most one equation each: the deterministic policy gradient on one slide, and the compatible critic plus \(\theta \leftarrow \theta + \alpha w\) on the next.

## Working agreement

Read `spec.md`, then this file, then the open task, then `code.md`.
Change the interface in `spec.md` and `code.md` together, in the same session that changes the code.
Do not start a task whose `Needs` line is still open.
Do not implement Soft Actor-Critic, the octopus arm, mountain car, puddle world, REDQ, or CrossQ.
