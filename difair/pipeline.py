"""Attribution of an outcome disparity to the pipeline stage that produced it.

Neither psychometric DIF analysis nor algorithmic fairness auditing answers the
question practitioners actually face: *given a group gap in the decisions a
system makes, which stage of the pipeline produced it?* Item-level DIF looks
only at the instrument; model-level fairness metrics look only at the
predictor. A disparity observed at the end of an assessment-to-decision
pipeline may originate at any of several stages, and the mitigation that helps
depends entirely on which one.

This module treats each stage as a player in a cooperative game. The
characteristic function is the disparity that remains once the mitigations for
a given subset of stages have been applied, and the Shapley value of a stage is
its average marginal reduction of that disparity across all orderings. Because
the stage count is small, the Shapley values are computed exactly by
enumerating all :math:`2^k` coalitions rather than sampled.

Stages and their mitigations
----------------------------
``item``
    Drop items flagged for DIF from the score.
``sampling``
    Reweight the training sample so groups are equally represented.
``model``
    Refit without features that act as group proxies.
``decision``
    Replace the single global cut score with group-calibrated thresholds.

Note on the ``decision`` stage
------------------------------
Group-calibrated thresholds equalise selection rates by construction, so with
``metric="demographic_parity"`` that mitigation drives the disparity to zero
whatever the other stages have done. Including ``decision`` therefore answers
"where can this gap be patched?", not "where did it come from?". For questions
of origin, use the default stage set, which omits ``decision``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from difair.fairness import demographic_parity, equal_opportunity

__all__ = ["AttributionResult", "attribute_stages", "DEFAULT_STAGES"]

DEFAULT_STAGES = ("item", "sampling", "model")
ALL_STAGES = ("item", "sampling", "model", "decision")


@dataclass
class AttributionResult:
    """Shapley attribution of a disparity to pipeline stages."""

    shapley: pd.DataFrame
    baseline_gap: float
    residual_gap: float
    metric: str
    coalition_values: dict
    response_kind: str = "dichotomous"

    @property
    def explained(self):
        """Total disparity removed once every mitigation is applied."""
        return self.baseline_gap - self.residual_gap

    def __repr__(self):  # pragma: no cover - display helper
        return (
            f"<AttributionResult metric={self.metric!r} "
            f"kind={self.response_kind!r} "
            f"baseline={self.baseline_gap:.4f} residual={self.residual_gap:.4f}>"
        )

    def summary(self, min_gap=0.02):
        """Shapley values with each stage's share of the explained disparity.

        Parameters
        ----------
        min_gap : float, default 0.02
            Shares are reported only when the explained disparity exceeds this
            value. Below it the ratio is a quotient of two near-zero quantities
            and carries no information, so ``share`` is returned as ``NaN``.
            The absolute Shapley values remain valid and should be read
            instead.
        """
        out = self.shapley.copy()
        total = self.explained
        if abs(total) > max(min_gap, 1e-12):
            out["share"] = out.shapley_value / total
        else:
            out["share"] = np.nan
        return out.sort_values("shapley_value", ascending=False).reset_index(drop=True)


def _build_score(responses, drop_items):
    """Summed score over the retained items.

    Summation is the right aggregation for dichotomous and ordered categorical
    responses alike: with 0/1 coding it is the number correct, and with ordered
    category codes it is the total rating, which is the matching criterion both
    families of DIF procedures already use.
    """
    keep = [c for c in responses.columns if c not in set(drop_items)]
    if not keep:
        raise ValueError(
            "The item mitigation would remove every item from the score: all "
            f"{responses.shape[1]} items are flagged. Purify the matching "
            "score before flagging, or raise the flagging threshold; with no "
            "item retained there is no instrument left to attribute."
        )
    return responses[keep].to_numpy(dtype=float).sum(axis=1)


def _response_kind(responses):
    """Classify a response matrix as dichotomous or polytomous.

    Returns ``"dichotomous"`` when every observed value is 0 or 1 and
    ``"polytomous"`` when the codes are integers spanning more than two
    categories. Anything else raises, because neither family of procedures can
    interpret continuous responses.
    """
    u = np.asarray(pd.DataFrame(responses).to_numpy(), dtype=float)
    obs = u[~np.isnan(u)]
    if obs.size == 0:
        raise ValueError("`responses` contains no observed values.")
    if not np.allclose(obs, np.round(obs)):
        raise ValueError(
            "`responses` must hold integer category codes; continuous values "
            "cannot be scored by either DIF family."
        )
    return "dichotomous" if np.all(np.isin(obs, [0.0, 1.0])) else "polytomous"


def _fit_predict(score, proxy, y, train_mask, group, use_proxy, balance, seed):
    """Fit the downstream model and return risk scores for everyone."""
    X = np.column_stack([score, proxy]) if use_proxy else score[:, None]
    Xtr, ytr = X[train_mask], y[train_mask]

    if len(np.unique(ytr)) < 2:
        return np.full(len(y), float(ytr.mean() if len(ytr) else 0.5))

    w = None
    if balance:
        gtr = group[train_mask]
        counts = pd.Series(gtr).value_counts()
        w = np.array([len(gtr) / (len(counts) * counts[gg]) for gg in gtr])

    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit((Xtr - mu) / sd, ytr, sample_weight=w)
    return clf.predict_proba((X - mu) / sd)[:, 1]


def _threshold(risk, group, rate, group_specific):
    """Convert risk scores to binary decisions at a target selection rate."""
    if not group_specific:
        cut = np.quantile(risk, 1.0 - rate)
        return (risk >= cut).astype(int)

    pred = np.zeros(len(risk), dtype=int)
    for lab in pd.unique(pd.Series(group)):
        m = group == lab
        if m.sum() == 0:
            continue
        cut = np.quantile(risk[m], 1.0 - rate)
        pred[m] = (risk[m] >= cut).astype(int)
    return pred


def attribute_stages(
    responses,
    group,
    focal_label,
    outcome,
    dif_items,
    proxy=None,
    train_mask=None,
    train_outcome=None,
    stages=DEFAULT_STAGES,
    metric="demographic_parity",
    selection_rate=0.4,
    seed=0,
):
    """Attribute a decision-stage disparity to the pipeline stages that caused it.

    Parameters
    ----------
    responses : DataFrame, shape (n_persons, n_items)
        Item responses forming the assessment. Dichotomous 0/1 codes and
        ordered category codes are both accepted; the stage score is the item
        sum in either case. Continuous values are rejected.
    group : array-like
        Binary group membership.
    focal_label : hashable
        Value of ``group`` identifying the focal group.
    outcome : array-like
        Binary ground-truth outcome the downstream model predicts.
    dif_items : sequence of str
        Items flagged by a DIF procedure: :func:`difair.dif.detect_dif` for
        dichotomous items or :func:`difair.poly.detect_dif_poly` for ordered
        ones. Both return a ``DIFResult`` whose ``.flagged`` attribute is
        accepted here unchanged.
    proxy : array-like, optional
        An auxiliary feature that may encode group membership. Required for the
        ``model`` stage; that stage is skipped when ``proxy`` is ``None``.
    train_mask : array-like of bool, optional
        Rows used to fit the model. Defaults to all rows, in which case the
        ``sampling`` stage has nothing to correct.
    train_outcome : array-like, optional
        Labels the model is trained on, which may differ from ``outcome`` when
        the historical record is itself biased. Evaluation always uses
        ``outcome``. Defaults to ``outcome``.
    stages : sequence of str, default ``("item", "sampling", "model")``
        Stages to include. See the module docstring on ``"decision"``.
    metric : {"demographic_parity", "equal_opportunity"}
        Disparity measure serving as the characteristic function.
    selection_rate : float, default 0.4
        Target proportion flagged by the decision rule.
    seed : int, default 0
        Seed passed to the downstream classifier.

    Returns
    -------
    AttributionResult
        ``.response_kind`` records whether the instrument was read as
        dichotomous or polytomous.

    Examples
    --------
    >>> from difair.simulate import simulate_pipeline_data
    >>> from difair.pipeline import attribute_stages
    >>> d = simulate_pipeline_data(seed=0)
    >>> res = attribute_stages(
    ...     d["responses"], d["group"], "focal", d["outcome"],
    ...     dif_items=d["dif_items"], proxy=d["proxy"],
    ...     train_mask=d["train_mask"], train_outcome=d["outcome_observed"],
    ... )
    >>> res.summary().columns.tolist()
    ['stage', 'shapley_value', 'share']
    """
    responses = pd.DataFrame(responses)
    kind = _response_kind(responses)  # validates the coding; raises if continuous
    group = np.asarray(pd.Series(group).to_numpy())
    outcome = np.asarray(outcome).astype(int)
    train_outcome = (
        outcome if train_outcome is None else np.asarray(train_outcome).astype(int)
    )
    n = len(group)

    if proxy is None:
        proxy = np.zeros(n)
        stages = tuple(s for s in stages if s != "model")
    else:
        proxy = np.asarray(proxy, dtype=float)

    if train_mask is None:
        train_mask = np.ones(n, dtype=bool)
        stages = tuple(s for s in stages if s != "sampling")
    else:
        train_mask = np.asarray(train_mask, dtype=bool)

    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        raise ValueError(f"Unknown stage(s): {sorted(unknown)}; allowed {ALL_STAGES}.")
    if metric not in ("demographic_parity", "equal_opportunity"):
        raise ValueError("`metric` must be 'demographic_parity' or 'equal_opportunity'.")
    if not stages:
        raise ValueError("No stage remains to attribute; supply `proxy` or `train_mask`.")

    stages = tuple(stages)
    cache: dict[frozenset, float] = {}

    def v(subset):
        """Absolute disparity remaining after applying the mitigations in ``subset``."""
        key = frozenset(subset)
        if key in cache:
            return cache[key]

        score = _build_score(responses, dif_items if "item" in key else ())
        risk = _fit_predict(
            score, proxy, train_outcome, train_mask, group,
            use_proxy="model" not in key,
            balance="sampling" in key,
            seed=seed,
        )
        pred = _threshold(risk, group, selection_rate, group_specific="decision" in key)

        if metric == "demographic_parity":
            val = demographic_parity(pred, group, focal_label)["difference"]
        else:
            val = equal_opportunity(outcome, pred, group, focal_label)["tpr_difference"]

        val = abs(val) if np.isfinite(val) else np.nan
        cache[key] = val
        return val

    k = len(stages)
    shapley = {s: 0.0 for s in stages}
    for i, stage in enumerate(stages):
        others = [s for s in stages if s != stage]
        for size in range(len(others) + 1):
            weight = factorial(size) * factorial(k - size - 1) / factorial(k)
            for combo in combinations(others, size):
                shapley[stage] += weight * (v(combo) - v(tuple(combo) + (stage,)))

    table = pd.DataFrame(
        {"stage": list(shapley), "shapley_value": [shapley[s] for s in shapley]}
    )
    return AttributionResult(
        shapley=table,
        baseline_gap=v(()),
        residual_gap=v(stages),
        metric=metric,
        coalition_values={tuple(sorted(k_)): val for k_, val in cache.items()},
        response_kind=kind,
    )
