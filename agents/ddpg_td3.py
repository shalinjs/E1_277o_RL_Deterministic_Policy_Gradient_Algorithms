"""DDPG and TD3 -- one neural actor-critic class, TD3 as three flags.

DDPG sets twin critics, delayed actor updates, and target smoothing off.
TD3 sets all three on (spec.md 1.4--1.5, code.md "DDPG and TD3"). The replay
buffer stays on the CPU. LunarLander is the environment this agent trains on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

AGENT_NAME_DDPG = "ddpg"
AGENT_NAME_TD3 = "td3"


@dataclass(frozen=True)
class TD3Flags:
    """The three differences between DDPG and TD3 (code.md).

    DDPG sets all three off: ``policy_delay == 1`` means the actor updates on every
    critic step. TD3 turns twin critics and target smoothing on and delays the
    actor.
    """

    twin: bool = False
    policy_delay: int = 1
    target_smoothing: bool = False

    @classmethod
    def ddpg(cls) -> "TD3Flags":
        return cls(twin=False, policy_delay=1, target_smoothing=False)

    @classmethod
    def td3(cls, policy_delay: int = 2) -> "TD3Flags":
        return cls(twin=True, policy_delay=policy_delay, target_smoothing=True)


class _MLP(nn.Module):
    """Two-hidden-layer MLP. The caller applies the output activation."""

    def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        width = in_dim
        for units in hidden:
            layers.append(nn.Linear(width, units))
            layers.append(nn.ReLU())
            width = units
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class Actor(_MLP):
    """tanh output, then an affine map onto ``[low, high]``."""

    def __init__(self, state_dim: int, action_dim: int, hidden: tuple[int, ...], low, high):
        super().__init__(state_dim, hidden, action_dim)
        self.register_buffer("low", torch.as_tensor(low, dtype=torch.float32))
        self.register_buffer("high", torch.as_tensor(high, dtype=torch.float32))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        unit = torch.tanh(super().forward(state))
        return self.low + (unit + 1.0) * 0.5 * (self.high - self.low)


class Critic(_MLP):
    """Q(s, a) from the concatenation of state and action."""

    def __init__(self, state_dim: int, action_dim: int, hidden: tuple[int, ...]):
        super().__init__(state_dim + action_dim, hidden, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return super().forward(torch.cat([state, action], dim=-1))


class ReplayBuffer:
    """Circular CPU buffer. A minibatch is sampled without replacement."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)
        self.size = 0
        self.pos = 0

    def add(self, state, action, reward, next_state, done: bool) -> None:
        self.state[self.pos] = state
        self.action[self.pos] = action
        self.reward[self.pos] = reward
        self.next_state[self.pos] = next_state
        self.done[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        index = rng.choice(self.size, size=batch_size, replace=False)
        return (
            self.state[index],
            self.action[index],
            self.reward[index],
            self.next_state[index],
            self.done[index],
        )


class DDPGTD3:
    """Shared DDPG/TD3 agent. ``flags`` is the only algorithmic difference."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        action_low,
        action_high,
        flags: TD3Flags,
        actor_lr: float,
        critic_lr: float,
        batch_size: int,
        buffer_size: int,
        tau: float,
        exploration_std: float,
        warmup_steps: int,
        gamma: float = 0.99,
        hidden: tuple[int, ...] = (256, 256),
        target_noise_std: float = 0.2,
        target_noise_clip: float = 0.5,
        seed: int = 0,
    ):
        if flags.policy_delay < 1:
            raise ValueError("policy_delay must be >= 1")
        self.flags = flags
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.actor_lr = float(actor_lr)
        self.critic_lr = float(critic_lr)
        self.batch_size = int(batch_size)
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.warmup_steps = int(warmup_steps)
        self.target_noise_std = float(target_noise_std)
        self.target_noise_clip = float(target_noise_clip)
        self.action_low = np.asarray(action_low, dtype=np.float32).reshape(self.action_dim)
        self.action_high = np.asarray(action_high, dtype=np.float32).reshape(self.action_dim)
        scale = np.maximum(np.abs(self.action_low), np.abs(self.action_high))
        # spec.md section 3: exploration_std is a fraction of the max absolute action.
        self.noise_std = np.float32(exploration_std) * scale
        self.rng = np.random.default_rng(seed)
        self.total_steps = 0
        self.critic_updates = 0

        self.actor = Actor(self.state_dim, self.action_dim, hidden, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_dim, self.action_dim, hidden, self.action_low, self.action_high)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.q1 = Critic(self.state_dim, self.action_dim, hidden)
        self.q1_target = Critic(self.state_dim, self.action_dim, hidden)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2 = None
        self.q2_target = None
        critic_params = list(self.q1.parameters())
        if flags.twin:
            self.q2 = Critic(self.state_dim, self.action_dim, hidden)
            self.q2_target = Critic(self.state_dim, self.action_dim, hidden)
            self.q2_target.load_state_dict(self.q2.state_dict())
            critic_params += list(self.q2.parameters())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_opt = torch.optim.Adam(critic_params, lr=self.critic_lr)
        self.buffer = ReplayBuffer(buffer_size, self.state_dim, self.action_dim)

    def act(self, state, explore: bool = True) -> np.ndarray:
        if explore and self.total_steps < self.warmup_steps:
            return self.rng.uniform(self.action_low, self.action_high).astype(np.float32)
        state_t = torch.as_tensor(np.asarray(state, dtype=np.float32)).reshape(1, self.state_dim)
        with torch.no_grad():
            action = self.actor(state_t).squeeze(0).cpu().numpy()
        if explore:
            action = action + self.rng.normal(0.0, self.noise_std).astype(np.float32)
            action = np.clip(action, self.action_low, self.action_high)
        return np.asarray(action, dtype=np.float32)

    def update(self, state, action, reward, next_state, done: bool) -> None:
        self.buffer.add(state, action, reward, next_state, done)
        self.total_steps += 1
        if self.total_steps <= self.warmup_steps or self.buffer.size < self.batch_size:
            return
        self._learn()

    def _tensors(self, *arrays):
        return tuple(torch.as_tensor(array, dtype=torch.float32) for array in arrays)

    def _q_target(self, reward, next_state, done) -> torch.Tensor:
        """Bootstrap target. ``done`` is a true termination, so it drops the next Q."""
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            if self.flags.target_smoothing:
                noise = torch.randn_like(next_action) * self.target_noise_std
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                low = torch.as_tensor(self.action_low)
                high = torch.as_tensor(self.action_high)
                next_action = (next_action + noise).clamp(low, high)
            next_q = self.q1_target(next_state, next_action)
            if self.flags.twin:
                next_q = torch.minimum(next_q, self.q2_target(next_state, next_action))
            return reward + (1.0 - done) * self.gamma * next_q

    def _learn(self) -> None:
        state, action, reward, next_state, done = self._tensors(
            *self.buffer.sample(self.batch_size, self.rng)
        )
        target = self._q_target(reward, next_state, done)
        loss = nn.functional.mse_loss(self.q1(state, action), target)
        if self.flags.twin:
            loss = loss + nn.functional.mse_loss(self.q2(state, action), target)
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()
        self.critic_updates += 1
        if self.critic_updates % self.flags.policy_delay != 0:
            return

        for parameter in self.q1.parameters():
            parameter.requires_grad_(False)
        # The deterministic gradient is evaluated at mu(s), not at the behaviour action.
        actor_loss = -self.q1(state, self.actor(state)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        for parameter in self.q1.parameters():
            parameter.requires_grad_(True)
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.q1, self.q1_target)
        if self.flags.twin:
            self._soft_update(self.q2, self.q2_target)

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for online, frozen in zip(source.parameters(), target.parameters()):
                frozen.mul_(1.0 - self.tau).add_(online, alpha=self.tau)

    def save(self, path: str) -> None:
        payload = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "q1": self.q1.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "total_steps": self.total_steps,
            "critic_updates": self.critic_updates,
            "twin": self.flags.twin,
        }
        if self.flags.twin:
            payload["q2"] = self.q2.state_dict()
            payload["q2_target"] = self.q2_target.state_dict()
        torch.save(payload, path)

    def load(self, path: str) -> None:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # older torch without weights_only
            payload = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(payload["actor"])
        self.actor_target.load_state_dict(payload["actor_target"])
        self.q1.load_state_dict(payload["q1"])
        self.q1_target.load_state_dict(payload["q1_target"])
        self.actor_opt.load_state_dict(payload["actor_opt"])
        self.critic_opt.load_state_dict(payload["critic_opt"])
        if self.flags.twin:
            self.q2.load_state_dict(payload["q2"])
            self.q2_target.load_state_dict(payload["q2_target"])
        self.total_steps = int(payload["total_steps"])
        self.critic_updates = int(payload["critic_updates"])


def _hidden(config: dict) -> tuple[int, ...]:
    if "hidden_dims" in config:
        dims = config["hidden_dims"]
        return tuple(int(unit) for unit in dims)
    if "hidden_size" in config:
        width = int(config["hidden_size"])
        return (width, width)
    return (256, 256)


def make_ddpg_td3(agent_name: str, env, config: dict) -> DDPGTD3:
    """Build DDPG or TD3. The agent name forces the three flags; a DDPG config cannot turn them on."""
    if agent_name == "ddpg":
        flags = TD3Flags.ddpg()
        target_noise_std, target_noise_clip = 0.2, 0.5
    elif agent_name == "td3":
        flags = TD3Flags.td3(policy_delay=int(config.get("policy_delay", 2)))
        target_noise_std = float(config.get("target_noise_std", 0.2))
        target_noise_clip = float(config.get("target_noise_clip", 0.5))
    else:
        raise ValueError(f"make_ddpg_td3 expected 'ddpg' or 'td3', got {agent_name!r}")
    if config.get("env") not in (None, "lunarlander"):
        raise ValueError("DDPG and TD3 run on LunarLander only")
    obs_space = env.observation_space
    act_space = env.action_space
    return DDPGTD3(
        int(np.prod(obs_space.shape)),
        int(np.prod(act_space.shape)),
        action_low=np.asarray(act_space.low, dtype=np.float32),
        action_high=np.asarray(act_space.high, dtype=np.float32),
        flags=flags,
        actor_lr=float(config["actor_lr"]),
        critic_lr=float(config["critic_lr"]),
        batch_size=int(config.get("batch_size", 256)),
        buffer_size=int(config.get("buffer_size", config.get("replay_capacity", 1_000_000))),
        tau=float(config.get("tau", 0.005)),
        exploration_std=float(config.get("exploration_std", 0.1)),
        warmup_steps=int(config.get("warmup_steps", 10_000)),
        gamma=float(config.get("gamma", 0.99)),
        hidden=_hidden(config),
        target_noise_std=target_noise_std,
        target_noise_clip=target_noise_clip,
        seed=int(config.get("seed", 0)),
    )
