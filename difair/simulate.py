"""Simulation of item responses with planted DIF.

Generates dichotomous responses from a 2PL item response model in which a
chosen subset of items carries DIF of a known magnitude. Because ground truth
is controlled by construction, these generators support the recovery studies
used to validate detection procedures: power, Type I error rate, and effect
size fidelity.

The two-parameter logistic model is

.. math::

    P(U_{ij} = 1 \\mid \\theta_i) = \\frac{1}{1 + \\exp(-a_j(\\theta_i - b_j))}

Uniform DIF shifts the difficulty ``b`` for the focal group; non-uniform DIF
shifts the discrimination ``a``. Group ability distributions may differ, which
lets impact be separated from DIF.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["SimulatedTest", "simulate_dif_data", "simulate_pipeline_data", "simulate_poly_dif_data", "simulate_poly_pipeline_data"]


@dataclass
class SimulatedTest:
    """Responses plus the generating parameters and planted DIF ground truth."""

    responses: pd.DataFrame
    group: np.ndarray
    theta: np.ndarray
    a_ref: np.ndarray
    b_ref: np.ndarray
    a_focal: np.ndarray
    b_focal: np.ndarray
    dif_items: list = field(default_factory=list)
    nonuniform_items: list = field(default_factory=list)

    @property
    def item_names(self):
        """Column labels of the simulated response matrix."""
        return list(self.responses.columns)

    def truth_frame(self):
        """Per-item ground truth, for scoring a detector's output."""
        names = self.item_names
        return pd.DataFrame(
            {
                "item": names,
                "has_dif": [nm in self.dif_items for nm in names],
                "has_nonuniform": [nm in self.nonuniform_items for nm in names],
                "b_shift": self.b_focal - self.b_ref,
                "a_shift": self.a_focal - self.a_ref,
            }
        )


def simulate_dif_data(
    n_ref=1000,
    n_focal=1000,
    n_items=30,
    n_dif_items=5,
    dif_magnitude=0.6,
    nonuniform_magnitude=0.0,
    impact=0.0,
    a_range=(0.7, 1.8),
    b_range=(-2.0, 2.0),
    seed=None,
):
    """Simulate a test with a known set of DIF items.

    Parameters
    ----------
    n_ref, n_focal : int
        Group sample sizes.
    n_items : int
        Test length.
    n_dif_items : int
        Number of items carrying planted DIF. The first ``n_dif_items`` items
        are selected so that ground truth is easy to inspect.
    dif_magnitude : float
        Difficulty shift ``b_focal - b_ref`` on DIF items. Positive values make
        the item harder for the focal group, i.e. DIF against the focal group.
    nonuniform_magnitude : float
        Discrimination shift ``a_focal - a_ref`` on DIF items. Non-zero values
        create non-uniform DIF.
    impact : float
        Mean ability difference (focal minus reference). Non-zero impact is a
        genuine group difference, not DIF, and a valid procedure must not
        confuse the two.
    seed : int, optional
        Seed for the random generator.

    Returns
    -------
    SimulatedTest
    """
    rng = np.random.default_rng(seed)

    if n_dif_items > n_items:
        raise ValueError("`n_dif_items` cannot exceed `n_items`.")

    a_ref = rng.uniform(*a_range, size=n_items)
    b_ref = rng.uniform(*b_range, size=n_items)
    a_focal, b_focal = a_ref.copy(), b_ref.copy()

    dif_idx = np.arange(n_dif_items)
    b_focal[dif_idx] += dif_magnitude
    a_focal[dif_idx] += nonuniform_magnitude

    theta_ref = rng.normal(0.0, 1.0, size=n_ref)
    theta_focal = rng.normal(impact, 1.0, size=n_focal)
    theta = np.concatenate([theta_ref, theta_focal])
    is_focal = np.concatenate([np.zeros(n_ref, bool), np.ones(n_focal, bool)])

    a = np.where(is_focal[:, None], a_focal[None, :], a_ref[None, :])
    b = np.where(is_focal[:, None], b_focal[None, :], b_ref[None, :])
    p = 1.0 / (1.0 + np.exp(-a * (theta[:, None] - b)))
    u = (rng.random(p.shape) < p).astype(int)

    names = [f"item_{j + 1:02d}" for j in range(n_items)]
    dif_names = [names[j] for j in dif_idx]
    nonuni = dif_names if nonuniform_magnitude != 0 else []

    return SimulatedTest(
        responses=pd.DataFrame(u, columns=names),
        group=np.where(is_focal, "focal", "reference"),
        theta=theta,
        a_ref=a_ref,
        b_ref=b_ref,
        a_focal=a_focal,
        b_focal=b_focal,
        dif_items=dif_names,
        nonuniform_items=nonuni,
    )


def simulate_pipeline_data(
    n_ref=1500,
    n_focal=1500,
    n_items=30,
    n_dif_items=6,
    dif_magnitude=0.8,
    proxy_strength=0.9,
    label_bias=0.35,
    undersample_focal=0.55,
    outcome_noise=0.5,
    seed=None,
):
    """Simulate an end-to-end assessment-to-decision pipeline with known causes.

    Provides the inputs :mod:`difair.pipeline` needs to attribute a decision
    gap to its originating stage. The design deliberately gives the two groups
    **identical ability distributions and identical true outcome rates**, so
    every disparity observed in the final decisions is an artefact of the
    pipeline rather than a real difference between groups. Three artefacts are
    planted independently:

    ``item``
        ``n_dif_items`` items are harder for the focal group by
        ``dif_magnitude`` logits, depressing focal scores.
    ``sampling``
        The focal group is retained in the training sample with probability
        ``undersample_focal``.
    ``model``
        Historical training labels discriminate: a fraction ``label_bias`` of
        genuinely positive focal cases are recorded as negative. A feature
        correlated with group at strength ``proxy_strength`` lets the model
        reproduce that pattern.

    Evaluation uses the *true* outcome, never the biased historical label.

    Returns
    -------
    dict
        ``responses``, ``group``, ``theta``, ``proxy``, ``outcome`` (true),
        ``outcome_observed`` (historically biased training label),
        ``dif_items``, ``train_mask``.
    """
    rng = np.random.default_rng(seed)

    sim = simulate_dif_data(
        n_ref=n_ref,
        n_focal=n_focal,
        n_items=n_items,
        n_dif_items=n_dif_items,
        dif_magnitude=dif_magnitude,
        impact=0.0,  # identical ability distributions by design
        seed=seed,
    )
    is_focal = sim.group == "focal"
    n = len(is_focal)

    # A feature correlated with group membership: the classic proxy variable.
    resid = max(1.0 - proxy_strength, 0.05)
    proxy = proxy_strength * is_focal.astype(float) + rng.normal(0, resid, n)

    # True outcome depends on ability only, so base rates are equal by group.
    latent = sim.theta + rng.normal(0, outcome_noise, n)
    outcome = (latent > np.quantile(latent, 0.6)).astype(int)

    # Historical label bias: some focal positives were recorded as negatives.
    observed = outcome.copy()
    flip = is_focal & (outcome == 1) & (rng.random(n) < label_bias)
    observed[flip] = 0

    # Under-represent the focal group in the training sample.
    keep_p = np.where(is_focal, undersample_focal, 1.0)
    train_mask = rng.random(n) < keep_p

    return {
        "responses": sim.responses,
        "group": sim.group,
        "theta": sim.theta,
        "proxy": proxy,
        "outcome": outcome,
        "outcome_observed": observed,
        "dif_items": sim.dif_items,
        "train_mask": train_mask,
    }


def simulate_poly_dif_data(
    n_ref=1000,
    n_focal=1000,
    n_items=20,
    n_categories=5,
    n_dif_items=4,
    dif_magnitude=0.6,
    impact=0.0,
    a_range=(0.7, 1.8),
    seed=None,
):
    """Simulate ordered categorical responses with planted DIF.

    Uses a generalised partial credit model: for an item with ``K`` categories
    and step difficulties ``d_1..d_{K-1}``,

    .. math::

        P(Y = k \\mid \\theta) \\propto
        \\exp\\!\\Big(\\sum_{m \\le k} a(\\theta - d_m)\\Big)

    Planted DIF shifts every step difficulty of the affected items by
    ``dif_magnitude`` for the focal group, so a positive value makes the item
    uniformly harder to endorse at higher categories, that is, DIF against the
    focal group.

    Parameters
    ----------
    n_categories : int, default 5
        Number of ordered response categories, coded ``0..n_categories - 1``.
        Five is the common Likert width.
    impact : float
        Mean latent-trait difference (focal minus reference). A genuine group
        difference, not DIF.

    Returns
    -------
    SimulatedTest
        ``responses`` holds integer category codes; ``b_ref`` and ``b_focal``
        hold the mean step difficulty per item.
    """
    rng = np.random.default_rng(seed)
    if n_dif_items > n_items:
        raise ValueError("`n_dif_items` cannot exceed `n_items`.")
    if n_categories < 3:
        raise ValueError("`n_categories` must be at least 3; use simulate_dif_data "
                         "for dichotomous items.")

    a = rng.uniform(*a_range, size=n_items)
    # Ordered step difficulties, spread around each item's location.
    loc = rng.uniform(-1.2, 1.2, size=n_items)
    offs = np.linspace(-1.0, 1.0, n_categories - 1)
    steps_ref = loc[:, None] + offs[None, :]
    steps_focal = steps_ref.copy()

    dif_idx = np.arange(n_dif_items)
    steps_focal[dif_idx] += dif_magnitude

    theta = np.concatenate([
        rng.normal(0.0, 1.0, size=n_ref),
        rng.normal(impact, 1.0, size=n_focal),
    ])
    is_focal = np.concatenate([np.zeros(n_ref, bool), np.ones(n_focal, bool)])

    u = np.empty((len(theta), n_items), dtype=int)
    for j in range(n_items):
        st = np.where(is_focal[:, None], steps_focal[j][None, :], steps_ref[j][None, :])
        # Cumulative sums of a(theta - d_m), with a leading zero for category 0.
        terms = a[j] * (theta[:, None] - st)
        num = np.concatenate([np.zeros((len(theta), 1)), np.cumsum(terms, axis=1)], axis=1)
        num -= num.max(axis=1, keepdims=True)          # stabilise the exponential
        p = np.exp(num)
        p /= p.sum(axis=1, keepdims=True)
        cdf = np.cumsum(p, axis=1)
        u[:, j] = (rng.random((len(theta), 1)) > cdf).sum(axis=1)

    names = [f"item_{j + 1:02d}" for j in range(n_items)]
    return SimulatedTest(
        responses=pd.DataFrame(u, columns=names),
        group=np.where(is_focal, "focal", "reference"),
        theta=theta,
        a_ref=a,
        b_ref=steps_ref.mean(axis=1),
        a_focal=a,
        b_focal=steps_focal.mean(axis=1),
        dif_items=[names[j] for j in dif_idx],
        nonuniform_items=[],
    )


def simulate_poly_pipeline_data(
    n_ref=1500,
    n_focal=1500,
    n_items=20,
    n_categories=5,
    n_dif_items=4,
    dif_magnitude=0.8,
    proxy_strength=0.9,
    label_bias=0.35,
    undersample_focal=0.55,
    outcome_noise=0.5,
    seed=None,
):
    """Pipeline simulation for ordered categorical instruments.

    The polytomous counterpart of :func:`simulate_pipeline_data`: a rating
    scale feeds a total score, the score feeds a predictive model, and the
    model feeds a decision. As in the dichotomous case the two groups are given
    identical latent distributions and identical true outcome rates, so every
    disparity in the final decisions is an artefact planted at a known stage.

    Returns
    -------
    dict
        ``responses`` (ordered category codes), ``group``, ``theta``,
        ``proxy``, ``outcome`` (true), ``outcome_observed`` (biased historical
        label), ``dif_items``, ``train_mask``.
    """
    rng = np.random.default_rng(seed)

    sim = simulate_poly_dif_data(
        n_ref=n_ref, n_focal=n_focal, n_items=n_items,
        n_categories=n_categories, n_dif_items=n_dif_items,
        dif_magnitude=dif_magnitude, impact=0.0, seed=seed,
    )
    is_focal = sim.group == "focal"
    n = len(is_focal)

    resid = max(1.0 - proxy_strength, 0.05)
    proxy = proxy_strength * is_focal.astype(float) + rng.normal(0, resid, n)

    latent = sim.theta + rng.normal(0, outcome_noise, n)
    outcome = (latent > np.quantile(latent, 0.6)).astype(int)

    observed = outcome.copy()
    flip = is_focal & (outcome == 1) & (rng.random(n) < label_bias)
    observed[flip] = 0

    keep_p = np.where(is_focal, undersample_focal, 1.0)
    train_mask = rng.random(n) < keep_p

    return {
        "responses": sim.responses,
        "group": sim.group,
        "theta": sim.theta,
        "proxy": proxy,
        "outcome": outcome,
        "outcome_observed": observed,
        "dif_items": sim.dif_items,
        "train_mask": train_mask,
    }
