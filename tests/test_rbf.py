"""T0.4: RBF features have a stable dimension, are finite, carry a bias of 1, and
two nearby states map to similar vectors.
"""

import numpy as np
import pytest

from features.rbf import feature_dim, pendulum_phi


@pytest.mark.parametrize("k", [5, 6, 7])
def test_feature_dimension_is_k_cubed_plus_bias(k):
    phi = pendulum_phi(np.array([1.0, 0.0, 0.0]), centres_per_dim=k)
    assert phi.shape == (k**3 + 1,)
    assert phi.shape[0] == feature_dim(k)


def test_feature_vector_is_finite():
    phi = pendulum_phi(np.array([0.3, -0.95, 4.0]))
    assert np.all(np.isfinite(phi))


def test_bias_coordinate_is_one():
    phi = pendulum_phi(np.array([1.0, 0.0, 0.0]))
    assert phi[-1] == pytest.approx(1.0)


def test_nearby_states_are_more_similar_than_distant_states():
    base = pendulum_phi(np.array([1.0, 0.0, 0.0]))
    near = pendulum_phi(np.array([0.999, 0.044, 0.1]))
    far = pendulum_phi(np.array([-1.0, 0.0, 0.0]))
    assert np.linalg.norm(base - near) < np.linalg.norm(base - far)


def test_works_on_a_pendulum_reset_observation():
    # Pendulum reset values live on the unit circle for (cos, sin).
    phi = pendulum_phi(np.array([np.cos(0.1), np.sin(0.1), 0.0]))
    assert np.all(np.isfinite(phi))
    assert phi.shape == (feature_dim(6),)


def test_rejects_degenerate_centre_counts():
    with pytest.raises(ValueError):
        pendulum_phi(np.array([1.0, 0.0, 0.0]), centres_per_dim=1)
