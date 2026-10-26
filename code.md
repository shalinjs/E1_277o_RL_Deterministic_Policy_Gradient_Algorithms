# Code contract

How `spec.md` is implemented. If a formula here and a formula in the 2014 paper disagree, stop and fix this file. The paper wins.

## Language and style

Python 3.11 or newer. NumPy for the linear agents. PyTorch for DDPG and TD3 only.
Standard library and the packages in `requirements.txt`. No new dependency without a task note.

Modules stay small and importable. No training at import time.
Scripts use `argparse`. Configs are YAML, loaded once in `run.py`, and passed as a plain dict or a frozen dataclass. Agents do not read files themselves except in `save` and `load`.

Names:

- `stochastic_ac`, never `sac`.
- `copdac_q` for the class. Natural gradient is a constructor flag `natural_gradient: bool = False`.
- `DDPGTD3` is the single neural class. `td3_flags` is a small object with `twin`, `policy_delay`, and `target_smoothing`. DDPG sets all three off (`policy_delay` is then 1, so the actor updates every critic step). TD3 sets all three on.

Comments explain an equation number or a non-obvious shape. They do not narrate an assignment.

## Seeds

One function, called from `run.py` before the environment is constructed:

```python
def set_seed(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
```

The bandit's orthogonal matrix and diagonal are drawn from `numpy.random.default_rng(seed)` after `set_seed`, so a seed fully determines `C`. Gymnasium is constructed with that same seed. Do not draw `C` from a second, unseeded generator.

## Environments

`envs/bandit.py` exposes a Gymnasium-like object:

- `observation_space`: a `Box` of shape `(1,)` whose only value is `0`. Agents must not treat that constant as a feature with a learned weight that changes the action, except as an explicit bias term already required by the spec.
- `action_space`: `Box(-inf, inf, shape=(m,))`. There is no clip on the bandit. The cost is defined for every real vector.
- `reset(seed) -> (obs, info)`, `step(a) -> (obs, reward, terminated, truncated, info)`.
- `terminated` is true after `max_steps` (default 10,000). `truncated` is always false.
- `info["mean_cost"]` is the cost of the current policy mean, supplied by the runner rather than the env. The env itself only knows the action it was given. The runner evaluates the mean action in a separate call and writes that cost to the CSV. Do not score the exploratory action in `eval_return`.
- `C` is built as `Q @ diag(d) @ Q.T` with `d[i]` in `{0.1, 1}` and `Q` from the QR decomposition of a Gaussian matrix, with signs fixed so the diagonal of `R` is positive.

Pendulum and LunarLander are constructed in `run.py` via `gymnasium.make`. A thin wrapper clips actions to `action_space` before `step`, for every agent. The wrapper does not change rewards.

Time-limit truncation: pass `done = terminated` into `update`. A `truncated` transition still bootstraps.

## Features

`features/rbf.py`:

```python
def pendulum_phi(state: np.ndarray) -> np.ndarray:
    """Map Pendulum observation (cos, sin, angular velocity) to a fixed RBF vector."""
```

- Centres: a regular grid, 6 centres per dimension unless the config sets `centres_per_dim` to 5 or 7.
- Angular-velocity centres span the Pendulum observation bound for that coordinate.
- Width: median distance between neighbouring centres along one axis, shared across axes after each axis is scaled to about `[-1, 1]`.
- Include a constant `1` bias coordinate at the end.
- Pure function. No state, no fitting.

The bandit does not use this function. Its feature for a linear policy mean is the identity: `mu = theta`, shape `(m,)`.

## Agent interface

```python
class Agent(Protocol):
    def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray: ...
    def update(self, state, action, reward, next_state, done: bool) -> None: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

`act` returns shape `(m,)` float64 or float32, finite. `explore=False` returns the Gaussian mean or `mu_theta(s)`, with no noise added.
`update` may learn zero times or many times. It must not call the environment.

`save` writes one file (NumPy `.npz` for linear agents, `torch.save` for the neural agent). `load` restores every learned array, including target networks and the log-standard-deviation. It does not restore the replay buffer.

## Runner and logs

`run.py` owns the loop:

1. Parse arguments and load YAML.
2. `set_seed`.
3. Build env and agent.
4. For each step: `act`, clip, `step`, `update`.
5. On the evaluation schedule, call `act(..., explore=False)` for the bandit mean cost, or run the configured number of evaluation episodes (default 5) for Gymnasium envs with exploration off.
6. Append one row and flush.

CSV path: `results/<env>/<agent>/seed<k>.csv`.
Header starts with `step,eval_return,wall_time_sec`. `wall_time_sec` is seconds since the start of the run, measured with `time.perf_counter`.
Optional later columns, only if the agent returned them: `q_estimate`, `rho_max`. Absent is fine. Present means finite floats.

Evaluation schedule: every `eval_every` steps from the config. Default 5,000. Bandit configs set `eval_every` to `2m`.

The runner writes the resolved config to `results/<env>/<agent>/seed<k>.yaml` so the YAML that was actually used sits next to the CSV.

## Config keys

Required in every YAML: `agent`, `env`, `seed`, `steps`, `gamma`, `eval_every`.

Bandit also requires `action_dim`.
Linear agents also require `actor_lr`, and either `critic_lr` or, for the bandit batch critic, nothing beyond the batch regression (no step size). Use `critic_lr` on Pendulum.
COPDAC and OffPAC require `sigma_beta`.
OffPAC may set `rho_clip: 10` or omit it. Omitting means no clip.
DDPG and TD3 require the table in `spec.md` section 3: `actor_lr`, `critic_lr`, `batch_size`, `buffer_size`, `tau`, `exploration_std`, `warmup_steps`, `hidden_size`.
TD3 adds `policy_delay`, `target_noise_std`, `target_noise_clip`. A DDPG YAML must not set those; the constructor forces them off.

## Updates to implement literally

Shapes are for the batch of one transition unless noted. `gamma` is 0.99 on Pendulum and LunarLander. The bandit is a one-step cost; the batch critic regresses onto the immediate cost, not a return.

### Stochastic Actor-Critic, bandit

Policy `Normal(theta, exp(y) * I)`. Compatible features for the joint parameter `(theta, y)`, plus a bias column of ones.
Every `2m` steps, solve least squares for the critic weights on the batch. One actor step uses that critic as `Q` in the stochastic policy gradient, averaged over the batch.
Score for the mean: `(a - theta) / var`. Implement the variance derivative with the same care: it is the derivative of `log pi` with respect to `y`, not a finite difference.

### Stochastic Actor-Critic, Pendulum

```
delta = r + gamma * V(next) * (1 - done) - V(state)
v <- v + critic_lr * delta * phi(state)
theta <- theta + actor_lr * delta * grad_theta log pi
y <- y + actor_lr * delta * grad_y log pi
```

`log_std` is clamped to a wide finite interval, for example `[-20, 2]`, so `exp(y)` cannot overflow. Record the clamp in the module docstring. This is a numerical guard, not a change of algorithm.

### OffPAC

Behaviour samples `a ~ Normal(mu_theta(s), sigma_beta^2 I)`.
Target density `pi` is the Gaussian whose mean and log-std are the learned actor, evaluated at the behaviour action.

```
rho = pi(a|s) / beta(a|s)
delta = r + gamma * V(next) * (1 - done) - V(state)
v <- v + critic_lr * rho * delta * phi(state)
theta <- theta + actor_lr * rho * delta * grad_theta log pi(a|s)
y <- y + actor_lr * rho * delta * grad_y log pi(a|s)
```

If `rho_clip` is set, clip `rho` into `[0, rho_clip]` after computing it, and count how many steps clipped. A run that clips on more than a few percent of steps is a result to report, not a silent success.

### COPDAC-Q, bandit

`mu = theta`, Jacobian is the identity.
Features for the regression: `(a - theta)` and a bias column. Least squares onto the observed costs every `2m` steps.
Actor, once per batch, using the fitted `w`:

```
theta <- theta + actor_lr * w          # if natural_gradient
theta <- theta + actor_lr * (Jacobian @ (Jacobian.T @ w))   # otherwise
```

With the identity Jacobian the ordinary update is `theta <- theta + actor_lr * w`. On the bandit the two flags therefore coincide when the Jacobian is `I`. That is correct. The flag still has to change the Pendulum update, where the Jacobian is `phi(s)` and the ordinary update is `phi(s) * (phi(s)^T w)`.

Do not multiply by an importance ratio.

### COPDAC-Q, Pendulum

`mu(s) = theta^T phi(s)` with `theta` of shape `(n, action_dim)`. Pendulum's action dimension is 1, so `theta` is `(n, 1)` and the Jacobian `grad_theta mu` is `phi(s)` reshaped to `(n, 1)`.

```
phi_sa = jacobian @ (a - mu)          # shape (n,)
Q = dot(phi_sa, w) + dot(phi(s), v)
delta = r + gamma * Q(next, mu(next)) * (1 - done) - Q
w <- w + critic_lr * delta * phi_sa
v <- v + critic_lr * delta * phi(s)
```

Actor, flag off: `theta <- theta + actor_lr * jacobian @ (jacobian.T @ w)`, with the product shaped so `theta` stays `(n, 1)`.
Actor, flag on: `theta <- theta + actor_lr * w`, with `w` shaped like `theta`.

The advantage of `a = mu(s)` is zero. The bootstrap in the TD target uses `mu(next)`, not the behaviour action. This is the whole of the off-policy deterministic update.

### DDPG and TD3

Networks: MLP, two hidden layers of 256, ReLU. Actor output `tanh`, then scaled affinely onto `[low, high]` of the action space. Critic inputs are the concatenation of state and action. Twin critics are two modules with independent initialisation, used only when `twin` is true.

Exploration noise is added after the squash, then clipped to `[low, high]`.
Warmup: for the first `warmup_steps`, `act(..., explore=True)` returns uniform random legal actions and `update` only stores transitions.

Replay: a circular buffer of `buffer_size`. Sample without replacement inside a minibatch.

Target soft update, after the steps that the delay allows:

```
p' <- (1 - tau) * p' + tau * p
```

for every actor and critic parameter.

TD3 bootstrap action: `clip(mu'(next) + clip(eps, -c, c), low, high)` with `eps ~ Normal(0, target_noise_std^2)`.
TD3 target value: `min(Q1', Q2')` at that action. DDPG uses the single target critic at `mu'(next)` with no added noise.

Actor loss: minimise `-Q1(s, mu_theta(s)).mean()` so the optimiser does gradient descent. That is ascent on `Q`. Do not also apply a separate manual gradient step.

Optimisers: Adam, learning rates from the config, created once in the constructor.

## Plots

`plots/make_plots.py` takes a results directory and writes PNGs and one markdown or CSV table. It groups by agent and environment, aggregates seeds, and draws mean ± standard deviation. It never fills a missing seed with a zero. A missing seed is a gap in the table.

Bandit y-axis is cost (lower is better). Gymnasium y-axis is return (higher is better). The axis label must say which.

## What not to do

- Do not put the replay buffer on the GPU.
- Do not call `loss.backward()` on a graph that still holds the target network. Target values are detached.
- Do not update the actor on the behaviour action. The deterministic gradient is evaluated at `mu(s)`.
- Do not share one `theta` tensor between the linear agents and the neural agent.
- Do not write credentials, hostnames of private machines, or absolute home-directory paths into configs. Relative paths only.
- Do not add a cryptographic hash of the config. A copied YAML next to the CSV is the record.

## Review checklist

Used when working T1.4.

- `explore=False` adds no noise.
- Bandit `eval_return` is the mean-action cost.
- COPDAC and DDPG actor updates do not contain `pi / beta`.
- OffPAC actor and critic both contain `rho`.
- TD3 and DDPG differ only by the three flags and the resulting control flow.
- `done` ignores time-limit truncation.
- A seeded bandit rebuilds the same `C`.
