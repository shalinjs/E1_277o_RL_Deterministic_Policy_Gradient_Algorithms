# Specification

Deterministic Policy Gradient Algorithms — E1 277 Reinforcement Learning.
Status: agreed from the 24 Oct 2026 project plan. Experiments freeze Thu 5 Nov 2026. Presentation Sun 15 Nov 2026.

The story the experiments must be able to tell: each algorithm fixes a weakness of the previous one.

1. Stochastic Actor-Critic (Gaussian) integrates over actions and gets worse as the action dimension grows.
2. COPDAC-Q replaces that integral by the deterministic policy gradient. The gap should grow with dimension.
3. DDPG keeps that gradient and replaces the linear critic and features by networks, a replay buffer, and target networks.
4. TD3 keeps DDPG and limits critic overestimation.

## 1. Algorithms in scope

Three main algorithms and three add-ons. An add-on is built only after its owner's main algorithm passes the sanity check in section 6. If time slips, the add-on is dropped before the main algorithm. TD3 is the add-on that is kept longest.

| Name | Owner | Kind | What it must show |
| --- | --- | --- | --- |
| Stochastic Actor-Critic (Gaussian) | A | main | On-policy Gaussian actor-critic. The baseline the 2014 paper beats on the bandit. |
| OffPAC | A | add-on | Same features and Gaussian target, but data from a fixed-width behaviour policy, reweighted by importance sampling. Separates "off-policy" from "deterministic". |
| COPDAC-Q | B | main | Off-policy deterministic actor-critic with a compatible linear critic. Silver et al. 2014, equations 19–22. |
| Natural-gradient COPDAC | B | add-on | Same agent as COPDAC-Q. Actor update replaced by `theta <- theta + alpha_theta * w`. End of section 4.3. |
| DDPG | C | main | Deterministic gradient with neural actor and critic. Lillicrap et al. 2016. |
| TD3 | C | add-on | DDPG plus clipped double Q, delayed actor updates, and target-policy smoothing. Fujimoto et al. 2018. Implemented as flags on the DDPG agent, not a second codebase. |

Naming rule: the string "SAC" is forbidden in code, configs, plot legends, and filenames. The 2018 Soft Actor-Critic algorithm is not this project.

### 1.1 Reward index

The 2014 paper writes `r_t = r(s_t, a_t)`. Gymnasium returns the reward after `step(a_t)`, which is the same quantity. Store it as the reward of the transition that produced it. Do not shift it by one the way Sutton and Barto's `R_{t+1}` notation might suggest.

### 1.2 Stochastic Actor-Critic (Gaussian)

Bandit (paper section 5.1):

- Policy `a ~ Normal(theta, exp(y))`, isotropic. Both the mean `theta` and the log-variance `y` are learned.
- Critic is least squares from compatible features onto the observed costs, refit every batch of `2m` steps. Features are `grad_theta log pi` and `grad_y log pi`, plus a bias feature.
- Actor takes one update per batch.
- A control with frozen variance is not required.

Pendulum (paper section 5.2, linear features):

- Mean `mu(s) = theta^T phi(s)`. Log standard deviation `log_std(s) = y^T phi(s)`.
- Critic `V(s) = v^T phi(s)`, updated by TD(0).
- Actor update uses the TD error as the advantage: step along `grad log pi(a|s) * delta`.
- Initial standard deviation is about half the legal action range.

### 1.3 OffPAC

Paper section 2.4 and the section 5.2 experimental setting. Degris, White, Sutton 2012, reduced as the 2014 paper reduces it.

- Target policy is the same Gaussian family as Stochastic Actor-Critic.
- Behaviour policy is the COPDAC behaviour: mean `mu_theta(s)` plus fixed-width Gaussian noise `sigma_beta`. On the bandit, `mu = theta`.
- Importance ratio `rho = pi(a|s) / beta(a|s)`, applied to both actor and critic.
- Critic `V(s) = v^T phi(s)`. The paper's OffPAC in section 5.2 uses gradient TD with `lambda = 0` and `alpha_u = 0`, which is ordinary off-policy TD(0). Implement that, not GTD2 and not emphatic TD.
- Actor step is `rho * delta * grad log pi(a|s)`.
- If `rho` produces NaN or enormous updates, clip it at 10 and record the clip in the config. Do not clip silently.

### 1.4 COPDAC-Q

Paper equations 19–22. Off-policy. No importance ratio on the actor or the critic.

Target `mu_theta(s) = theta^T phi(s)`. On the bandit, `phi` is the identity and `mu = theta`.
Behaviour `a ~ Normal(mu_theta(s), sigma_beta^2 I)`.

Compatible critic

```
Q(s, a) = (a - mu_theta(s))^T (grad_theta mu_theta(s))^T w  +  V(s)
V(s) = v^T phi(s)
```

On the bandit the centred feature is `(a - theta)`, plus a bias feature, and the critic is least squares onto the costs, refit every `2m` steps, one actor step per batch. This matches section 5.1, where the critic is batch regression rather than the incremental update.

On Pendulum use the incremental updates:

```
delta = r + gamma * Q(s', mu(s')) * (1 - done) - Q(s, a)
theta <- theta + alpha_theta * grad_theta mu(s) * (grad_theta mu(s)^T w)
w <- w + alpha_w * delta * phi(s, a)
v <- v + alpha_v * delta * phi(s)
```

`done` is true only when the environment terminates, not when it truncates at a time limit. Bootstrapping through a time-limit cut is required.

Natural-gradient flag: the actor line becomes `theta <- theta + alpha_theta * w`. Critic updates stay the same. Default flag is off.

### 1.5 DDPG

Deterministic actor `mu_theta(s)`, critic `Q_w(s, a)`.

- Behaviour: `mu_theta(s)` plus Gaussian exploration noise, clipped to the action range. No Ornstein–Uhlenbeck noise.
- Replay buffer. Learn from minibatches after a warmup of random actions.
- Target networks `theta'` and `w'`, soft update with rate `tau` after every learner step.
- Critic target `r + gamma * Q_w'(s', mu_theta'(s')) * (1 - done)`.
- Critic loss: mean squared error against that target.
- Actor update: ascend the deterministic policy gradient through the critic,
  `grad_theta mu(s) * grad_a Q_w(s, a)` evaluated at `a = mu_theta(s)`,
  averaged over the minibatch. Implemented as ascent on `Q_w(s, mu_theta(s))`, which is the same quantity.

### 1.6 TD3

Same code path as DDPG. Three flags, all off for a DDPG run and all on for a TD3 run. Do not change any other hyperparameter between the two.

1. Twin critics. Target is `r + gamma * min(Q1', Q2')` at the smoothed next action. Both critics learn. The actor uses only the first critic.
2. Delayed actor updates. Actor and target networks update every `policy_delay` critic updates. Default delay is 2.
3. Target policy smoothing. The bootstrap action is `mu_theta'(s') + noise`, with noise drawn from a clipped Gaussian. Defaults: standard deviation 0.2, clip 0.5, both in absolute action units, then clipped to the legal action range.

Optional diagnostic, not required for the sanity check: log the critic's predicted Q against the realised discounted return, so a later plot can show overestimation.

## 2. Environments

Three environments. Each answers one question. Pendulum is the only place every algorithm is expected to meet.

### 2.1 Continuous quadratic bandit

Paper section 5.1. Member A builds it. Members A and B run it. DDPG on the bandit is optional and is not required for the freeze.

- Action `a in R^m`, `m in {10, 25, 50}`. No state. The agent still sees a dummy state vector so the shared interface does not branch.
- Cost `(a - a_star)^T C (a - a_star)`, reward `= -cost`, `a_star = (4, ..., 4)`.
- `C = Q D Q^T`. `Q` is a random orthogonal matrix, fixed for a seed. `D` is diagonal with each entry drawn from `{0.1, 1}`, fixed for a seed.
- Policy mean starts at 0.
- Horizon: 10,000 environment steps.
- What is scored: the cost of the mean action, not the cost of the exploratory action.
- Question: does the deterministic update pull away from the stochastic one as `m` grows?

### 2.2 Pendulum-v1

Gymnasium. All six algorithms, except natural-gradient COPDAC is run here only if it helped on the bandit.

- Torque clipped to `[-2, 2]`.
- About 50,000 environment steps.
- Discount used by the agents: 0.99.
- Question: on a one-dimensional action, how do on-policy, off-policy, stochastic, deterministic, linear, and neural methods compare? A small gap between Stochastic Actor-Critic and COPDAC-Q is an expected result, not a bug. The paper's Figure 2 is the same observation.

### 2.3 LunarLanderContinuous-v3

Gymnasium Box2D. DDPG and TD3 only.

- About 300,000 environment steps. Fallback if a run exceeds 1.5 hours: 200,000 steps, stated in the config.
- Question: does TD3 reduce DDPG's instability on a harder task?
- If Box2D will not install, install `swig` first, then the env extra. Colab is an acceptable place to run the seeds. Do not replace LunarLander by another env.

## 3. Experiment rules

- Seeds `0, 1, 2, 3, 4` for every final cell. Report mean and standard deviation across seeds.
- Evaluate the mean or deterministic policy (`explore=False`) every 5,000 environment steps. On the bandit, evaluate once per batch of `2m` steps.
- Linear methods: before the 5-seed run, sweep 3 or 4 step sizes on seed 0 only. Keep the rest of the paper's protocol. Record the sweep and the chosen value in the config.
- DDPG and TD3 use the same starting values. Do not sweep them independently.

| Setting | Starting value |
| --- | --- |
| Actor and critic | 2 hidden layers, 256 units, ReLU, tanh output scaled to the action range |
| Learning rates | 3e-4 for actor and critic |
| Discount | 0.99 |
| Batch size | 256 |
| Replay capacity | 1,000,000 |
| Target rate tau | 0.005 |
| Exploration noise | Gaussian, standard deviation 0.1 times the max absolute action |
| Warmup random actions | 10,000 (Pendulum: 1,000 to 5,000) |
| TD3 policy delay | 2 |
| TD3 target noise / clip | 0.2 / 0.5 |

### Metrics

- Learning curve: bandit cost, or episodic return, against environment steps. One curve per algorithm, mean and a standard-deviation band.
- Sample efficiency: environment steps until Pendulum mean return exceeds −300, and until LunarLander mean return exceeds 200. If the threshold is never hit, say so. Do not invent a crossing.
- Wall-clock seconds per run, from the log.
- Stability: the seed spread. This is the comparison TD3 is for.
- Optional, bandit: variance of 100 gradient estimates at a fixed parameter, Stochastic Actor-Critic against COPDAC-Q, as a picture of section 6.
- Optional, TD3: predicted Q against realised return.

## 4. Shared interface

Every agent implements:

- `act(state, explore=True) -> action`. `explore=False` is the mean or deterministic action. No environment step inside `act`.
- `update(state, action, reward, next_state, done) -> None`. Called once per real environment step. DDPG and TD3 push the transition into replay and may learn inside this call. Linear bandit agents may no-op until the batch is full and then refit.
- `save(path)` and `load(path)`.

`run.py` is the only entry point for a training run:

```
python run.py --agent <name> --env <name> --seed <int> --steps <int> --config <yaml>
```

Agent names: `stochastic_ac`, `offpac`, `copdac_q`, `ddpg`, `td3`.
Environment names: `bandit`, `pendulum`, `lunarlander`.
Natural gradient is `agent: copdac_q` with `natural_gradient: true` in the config, not a sixth agent name. TD3 is a separate agent name so logs stay distinct, and it constructs the same class as DDPG with the three flags forced on.

One CSV per run, written only by the runner:

```
results/<env>/<agent>/seed<k>.csv
```

`<env>` is `bandit_m<m>` on the bandit, so the three action dimensions do not overwrite each other. `<agent>` is `copdac_q_natural` when `natural_gradient: true`, so the two COPDAC variants stay separate. Command-line `--seed` and `--steps` override the YAML; a YAML `agent` or `env` that disagrees with the command line is an error.

Columns, in order: `step`, `eval_return`, `wall_time_sec`. Extra columns are allowed after these three. `eval_return` is the bandit mean-action cost, or the mean return of evaluation episodes. The header is fixed. Nobody edits a CSV by hand.

Configs live in `configs/` as YAML, one file per agent and environment that is actually run. The file used for a final run is committed next to the result.

## 5. Repository layout

```
envs/bandit.py
features/rbf.py
agents/stochastic_ac.py
agents/offpac.py
agents/copdac_q.py
agents/ddpg_td3.py
run.py
configs/
results/
plots/make_plots.py
requirements.txt
```

`requirements.txt` pins a working set for `numpy`, `torch`, `gymnasium[box2d]`, `matplotlib`, `pyyaml`. Plots and the results table are produced only by `plots/make_plots.py`.

Pendulum features for the linear methods are RBFs on `(cos angle, sin angle, angular velocity)`, a grid of about 5 to 7 centres along each of the three inputs. Member B owns this function. Stochastic Actor-Critic and OffPAC on Pendulum call it. They do not invent a second feature map.

## 6. Acceptance

An algorithm is done when its sanity check passes on one seed. The experiment is done when all five seeds are logged and the plot script has drawn them.

By Thu 29 Oct:

- Stochastic Actor-Critic: on the bandit at `m = 10`, mean-action cost falls steadily and the mean moves toward 4.
- COPDAC-Q: same bandit, cost falls faster than Stochastic Actor-Critic, mean moves toward 4.
- DDPG: Pendulum return reaches about −200 or better within 50,000 steps. The environment's best return is 0.
- TD3: the same Pendulum bar, with all three flags on.
- Plot script: a learning curve with mean and standard-deviation band from two dummy CSVs.

By Sun 1 Nov, if the main checks passed:

- OffPAC and natural-gradient COPDAC each learn on at least one environment without diverging.
- Bandit curves exist for `m = 10, 25, 50`, five seeds, for Stochastic Actor-Critic, COPDAC-Q, and natural-gradient COPDAC.

By Thu 5 Nov: final plots and the results table are generated by the script.

## 7. Risks already decided

| What goes wrong | What to do |
| --- | --- |
| COPDAC-Q weights or Q become NaN | Drop the critic step size by 3 to 10 times. Mention COPDAC-GQ in the report. Do not implement COPDAC-GQ unless the spec is changed. |
| Linear methods stay near −1200 on Pendulum after 50,000 steps | On Mon 2 Nov, drop them from Pendulum and keep the bandit. Write that decision into this spec. |
| OffPAC ratios explode | Clip at 10 and record it, or run OffPAC only on Pendulum. |
| LunarLander is too slow | Parallel seeds, or cut to 200,000 steps in the config. |
| TD3 does not beat DDPG on Pendulum | Keep the result. Show seed spread and, if available, the Q-overestimation plot. LunarLander is where a gap is more likely. |

## 8. Report boundaries

These are not code tasks. They constrain what the logs must contain.

- Slides: 10 slides, 15 minutes. One main plot on each results slide.
- Backup slides, not counted in the 10: hyperparameter table, per-dimension bandit curves, TD3 ablation, OffPAC ratio statistics, compatible-critic derivation.
- Theory slides carry at most one equation each. The report holds the derivations.
- Report owners: A writes the introduction, B writes the 2014 theorem and the bandit results, C writes the experimental setup and leads the discussion, each member writes their own algorithm section, A and C share Pendulum and LunarLander results.
