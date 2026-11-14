"""DDPG and TD3 share one class. The three flags are the only algorithmic difference."""

import numpy as np
import torch

from agents.ddpg_td3 import TD3Flags, make_ddpg_td3


class _Space:
    def __init__(self, low, high, shape):
        self.low = np.full(shape, low, dtype=np.float32)
        self.high = np.full(shape, high, dtype=np.float32)
        self.shape = shape
        self.dtype = np.float32


class _Env:
    observation_space = _Space(-1.0, 1.0, (3,))
    action_space = _Space(-2.0, 2.0, (1,))


def _config(**extra):
    config = dict(
        env="lunarlander",
        seed=0,
        actor_lr=1e-2,
        critic_lr=1e-2,
        batch_size=4,
        buffer_size=50,
        tau=1.0,
        exploration_std=0.5,
        warmup_steps=0,
        hidden_size=8,
        gamma=0.5,
    )
    config.update(extra)
    return config


def _agent(name="ddpg", **extra):
    return make_ddpg_td3(name, _Env(), _config(**extra))


def _fill(agent, n, reward=1.0, done=False):
    state = np.zeros(agent.state_dim, dtype=np.float32)
    action = np.zeros(agent.action_dim, dtype=np.float32)
    for _ in range(n):
        agent.update(state, action, reward, state, done)


def test_explore_false_is_deterministic_and_inside_bounds():
    agent = _agent(warmup_steps=5)
    state = np.array([0.2, -0.3, 0.4], dtype=np.float32)
    first = agent.act(state, explore=False)
    second = agent.act(state, explore=False)
    np.testing.assert_allclose(first, second)
    assert np.all(first >= agent.action_low - 1e-5)
    assert np.all(first <= agent.action_high + 1e-5)


def test_warmup_actions_are_random_and_do_not_learn():
    agent = _agent(warmup_steps=3, exploration_std=0.0)
    before = [parameter.detach().clone() for parameter in agent.q1.parameters()]
    state = np.zeros(3, dtype=np.float32)
    samples = [agent.act(state, explore=True) for _ in range(2)]
    assert not np.allclose(samples[0], samples[1])
    _fill(agent, 3)
    for old, new in zip(before, agent.q1.parameters()):
        assert torch.equal(old, new)
    assert agent.critic_updates == 0


def test_ddpg_learns_on_the_first_full_batch_and_td3_delays_the_actor():
    ddpg = _agent("ddpg")
    td3 = _agent("td3", policy_delay=2)
    ddpg_actor = [parameter.detach().clone() for parameter in ddpg.actor.parameters()]
    td3_actor = [parameter.detach().clone() for parameter in td3.actor.parameters()]
    _fill(ddpg, 4)
    _fill(td3, 4)
    assert ddpg.critic_updates == 1 and td3.critic_updates == 1
    assert any(not torch.equal(old, new) for old, new in zip(ddpg_actor, ddpg.actor.parameters()))
    assert all(torch.equal(old, new) for old, new in zip(td3_actor, td3.actor.parameters()))
    held = [parameter.detach().clone() for parameter in td3.actor.parameters()]
    _fill(td3, 1)
    assert td3.critic_updates == 2
    assert any(not torch.equal(old, new) for old, new in zip(held, td3.actor.parameters()))


def test_agent_name_forces_the_three_flags():
    ddpg = _agent("ddpg", policy_delay=5, target_noise_std=0.9, target_noise_clip=0.4)
    td3 = _agent("td3")
    assert ddpg.flags == TD3Flags.ddpg()
    assert td3.flags == TD3Flags.td3(policy_delay=2)
    assert ddpg.q2 is None and td3.q2 is not None


def test_terminal_transition_drops_the_bootstrap():
    agent = _agent()
    for module in (agent.actor_target, agent.q1_target):
        for parameter in module.parameters():
            parameter.data.zero_()
    # Last linear bias is the constant Q. ReLU(0) into a zero-weight layer stays at that bias.
    agent.q1_target.net[-1].bias.data.fill_(4.0)
    reward = torch.ones(2, 1)
    next_state = torch.zeros(2, agent.state_dim)
    done = torch.tensor([[0.0], [1.0]])
    target = agent._q_target(reward, next_state, done)
    np.testing.assert_allclose(target[0].item(), 1.0 + 0.5 * 4.0)
    np.testing.assert_allclose(target[1].item(), 1.0)


def test_td3_target_uses_the_minimum_of_the_twin_critics():
    agent = _agent("td3")
    for module in (agent.actor_target, agent.q1_target, agent.q2_target):
        for parameter in module.parameters():
            parameter.data.zero_()
    agent.q1_target.net[-1].bias.data.fill_(4.0)
    agent.q2_target.net[-1].bias.data.fill_(1.5)
    reward = torch.zeros(1, 1)
    next_state = torch.zeros(1, agent.state_dim)
    done = torch.zeros(1, 1)
    target = agent._q_target(reward, next_state, done)
    np.testing.assert_allclose(target.item(), 0.5 * 1.5)


def test_save_load_restores_the_actor_and_not_the_replay(tmp_path):
    agent = _agent()
    _fill(agent, 6, reward=-0.3)
    path = str(tmp_path / "agent.pt")
    agent.save(path)
    fresh = _agent()
    fresh.load(path)
    state = np.linspace(-0.2, 0.2, agent.state_dim, dtype=np.float32)
    np.testing.assert_allclose(fresh.act(state, explore=False), agent.act(state, explore=False))
    assert fresh.buffer.size == 0
    assert fresh.total_steps == agent.total_steps


def test_factory_rejects_a_non_lunarlander_env():
    import pytest

    with pytest.raises(ValueError):
        make_ddpg_td3("ddpg", _Env(), _config(env="bandit"))
