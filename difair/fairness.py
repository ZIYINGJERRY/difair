"""Model-level algorithmic fairness metrics.

Complements :mod:`difair.dif`, which works at the item level. The functions
here operate on the predictions of a downstream model -- the at-risk flags,
placement decisions and automated scores that educational data pipelines
ultimately produce.

Every metric returns both the per-group values and the between-group
disparity, because a single scalar hides which group is disadvantaged and by
how much.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "group_rates",
    "demographic_parity",
    "equalized_odds",
    "equal_opportunity",
    "predictive_parity",
    "calibration_gap",
    "fairness_report",
    "ordinal_group_summary",
    "ordinal_disparity",
    "ordinal_disparity_replicate",
]


def _prep(y_true, y_pred, group, y_score=None):
    y_pred = np.asarray(y_pred).astype(float)
    g = np.asarray(pd.Series(group).to_numpy())
    y_true = None if y_true is None else np.asarray(y_true).astype(float)
    if y_score is not None:
        y_score = np.asarray(y_score).astype(float)

    n = len(y_pred)
    if len(g) != n:
        raise ValueError("`group` length does not match `y_pred`.")
    if y_true is not None and len(y_true) != n:
        raise ValueError("`y_true` length does not match `y_pred`.")
    return y_true, y_pred, g, y_score


def _safe_div(num, den):
    return float(num / den) if den > 0 else np.nan


def group_rates(y_true, y_pred, group):
    """Per-group confusion-matrix rates.

    Returns
    -------
    DataFrame
        One row per group with ``n``, ``selection_rate``, ``tpr``, ``fpr``,
        ``tnr``, ``fnr``, ``ppv`` and ``accuracy``.
    """
    y_true, y_pred, g, _ = _prep(y_true, y_pred, group)
    rows = []
    for lab in pd.unique(pd.Series(g)):
        m = g == lab
        yp = y_pred[m]
        row = {"group": lab, "n": int(m.sum()), "selection_rate": float(yp.mean())}
        if y_true is not None:
            yt = y_true[m]
            tp = float(((yp == 1) & (yt == 1)).sum())
            fp = float(((yp == 1) & (yt == 0)).sum())
            tn = float(((yp == 0) & (yt == 0)).sum())
            fn = float(((yp == 0) & (yt == 1)).sum())
            row.update(
                tpr=_safe_div(tp, tp + fn),
                fpr=_safe_div(fp, fp + tn),
                tnr=_safe_div(tn, tn + fp),
                fnr=_safe_div(fn, fn + tp),
                ppv=_safe_div(tp, tp + fp),
                accuracy=_safe_div(tp + tn, len(yt)),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _gap(rates, col, focal_label):
    """Focal minus reference on ``col``; negative means focal disadvantaged."""
    if col not in rates.columns:
        return np.nan
    foc = rates.loc[rates.group == focal_label, col]
    ref = rates.loc[rates.group != focal_label, col]
    if foc.empty or ref.empty:
        return np.nan
    return float(foc.iloc[0] - ref.iloc[0])


def demographic_parity(y_pred, group, focal_label):
    """Difference and ratio in selection rate between focal and reference."""
    rates = group_rates(None, y_pred, group)
    diff = _gap(rates, "selection_rate", focal_label)
    foc = float(rates.loc[rates.group == focal_label, "selection_rate"].iloc[0])
    ref = float(rates.loc[rates.group != focal_label, "selection_rate"].iloc[0])
    return {
        "metric": "demographic_parity",
        "focal_rate": foc,
        "reference_rate": ref,
        "difference": diff,
        "ratio": _safe_div(min(foc, ref), max(foc, ref)) if max(foc, ref) > 0 else np.nan,
    }


def equalized_odds(y_true, y_pred, group, focal_label):
    """TPR and FPR gaps; the equalized-odds violation is the larger absolute gap."""
    rates = group_rates(y_true, y_pred, group)
    tpr, fpr = _gap(rates, "tpr", focal_label), _gap(rates, "fpr", focal_label)
    finite = [abs(v) for v in (tpr, fpr) if np.isfinite(v)]
    return {
        "metric": "equalized_odds",
        "tpr_difference": tpr,
        "fpr_difference": fpr,
        "max_violation": max(finite) if finite else np.nan,
    }


def equal_opportunity(y_true, y_pred, group, focal_label):
    """TPR gap alone: equal opportunity restricted to the positive class."""
    rates = group_rates(y_true, y_pred, group)
    return {"metric": "equal_opportunity", "tpr_difference": _gap(rates, "tpr", focal_label)}


def predictive_parity(y_true, y_pred, group, focal_label):
    """Positive predictive value gap."""
    rates = group_rates(y_true, y_pred, group)
    return {"metric": "predictive_parity", "ppv_difference": _gap(rates, "ppv", focal_label)}


def calibration_gap(y_true, y_score, group, focal_label, n_bins=10):
    """Group-wise expected calibration error and its focal-reference gap.

    Scores are binned into ``n_bins`` equal-width bins on [0, 1]; within each
    bin the absolute difference between mean predicted probability and observed
    frequency is averaged, weighted by bin size.
    """
    y_true, _, g, y_score = _prep(y_true, np.zeros(len(y_score)), group, y_score)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = {}
    for lab in pd.unique(pd.Series(g)):
        m = g == lab
        s, t = y_score[m], y_true[m]
        if s.size == 0:
            out[lab] = np.nan
            continue
        idx = np.clip(np.digitize(s, edges[1:-1]), 0, n_bins - 1)
        ece = 0.0
        for b in range(n_bins):
            sel = idx == b
            if sel.sum() == 0:
                continue
            ece += (sel.sum() / s.size) * abs(s[sel].mean() - t[sel].mean())
        out[lab] = float(ece)

    ref_labels = [k for k in out if k != focal_label]
    ref = out[ref_labels[0]] if ref_labels else np.nan
    return {
        "metric": "calibration_gap",
        "focal_ece": out.get(focal_label, np.nan),
        "reference_ece": ref,
        "difference": float(out.get(focal_label, np.nan) - ref)
        if np.isfinite(out.get(focal_label, np.nan)) and np.isfinite(ref)
        else np.nan,
    }


def fairness_report(y_true, y_pred, group, focal_label, y_score=None):
    """Run every applicable metric and return a tidy summary table.

    Parameters
    ----------
    y_true : array-like or None
        Ground-truth labels. If ``None`` only demographic parity is computed.
    y_pred : array-like
        Binary decisions.
    y_score : array-like, optional
        Predicted probabilities, required for :func:`calibration_gap`.

    Returns
    -------
    DataFrame
        Columns ``metric``, ``value``, ``detail``. ``value`` is signed so that
        negative numbers indicate the focal group is disadvantaged.
    """
    rows = [demographic_parity(y_pred, group, focal_label)]
    if y_true is not None:
        rows += [
            equalized_odds(y_true, y_pred, group, focal_label),
            equal_opportunity(y_true, y_pred, group, focal_label),
            predictive_parity(y_true, y_pred, group, focal_label),
        ]
        if y_score is not None:
            rows.append(calibration_gap(y_true, y_score, group, focal_label))

    key = {
        "demographic_parity": "difference",
        "equalized_odds": "max_violation",
        "equal_opportunity": "tpr_difference",
        "predictive_parity": "ppv_difference",
        "calibration_gap": "difference",
    }
    tidy = []
    for r in rows:
        m = r["metric"]
        tidy.append(
            {
                "metric": m,
                "value": r.get(key[m], np.nan),
                "detail": {k: v for k, v in r.items() if k != "metric"},
            }
        )
    return pd.DataFrame(tidy)


# --------------------------------------------------------------------------- #
# ordinal outcomes
# --------------------------------------------------------------------------- #
def ordinal_group_summary(y_true, group):
    """Per-group distribution of an ordered outcome.

    Returns
    -------
    DataFrame
        One row per group with ``n``, ``mean_level``, and the proportion in
        each category.
    """
    y = np.asarray(y_true).astype(float)
    g = np.asarray(pd.Series(group).to_numpy())
    levels = np.unique(y[~np.isnan(y)])
    rows = []
    for lab in pd.unique(pd.Series(g)):
        m = (g == lab) & ~np.isnan(y)
        yy = y[m]
        row = {"group": lab, "n": int(m.sum()),
               "mean_level": float(yy.mean()) if yy.size else np.nan}
        for lv in levels:
            row[f"p_{int(lv)}"] = float((yy == lv).mean()) if yy.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _ps_point(foc, ref):
    """Probability that a random focal case outranks a random reference case."""
    ranks = stats.rankdata(np.concatenate([foc, ref]))
    m = foc.size
    return float((ranks[:m].sum() - m * (m + 1) / 2) / (m * ref.size))


def ordinal_disparity_replicate(
    y_true, group, focal_label, weights=None, replicate_weights=None,
    method="jackknife", fay_factor=0.5, fpc=None,
):
    """Probability of superiority with a design-based standard error.

    The Hanley-McNeil error reported by :func:`ordinal_disparity` assumes the
    two groups were sampled independently, which a clustered survey violates.
    This recomputes the statistic under each replicate weight vector and takes
    the variance from their spread, exactly as :func:`difair.survey.survey_dif`
    does for item-level statistics.

    Weighted ranks are handled by treating each respondent's weight as a
    replication count, so a replicate that zeroes a cluster removes it from
    the comparison entirely.

    Returns
    -------
    dict
        ``probability_superiority``, ``se``, ``ci_low``, ``ci_high``,
        ``n_replicates``, ``method``.
    """
    from difair.survey import replicate_variance

    y = np.asarray(y_true).astype(float)
    g = np.asarray(pd.Series(group).to_numpy())
    ok = ~np.isnan(y)
    y, g = y[ok], g[ok]
    w = np.ones(y.shape) if weights is None else np.asarray(weights, float)[ok]

    def ps_weighted(wv):
        keep = wv > 0
        if keep.sum() < 4:
            return np.nan
        yy, gg, ww = y[keep], g[keep], wv[keep]
        # Replicate each observation in proportion to its weight, rounded to a
        # bounded multiplier so the rank computation stays tractable.
        mult = np.maximum(1, np.round(ww / ww.mean() * 4).astype(int))
        idx = np.repeat(np.arange(len(yy)), mult)
        yr, gr = yy[idx], gg[idx]
        foc, ref = yr[gr == focal_label], yr[gr != focal_label]
        return _ps_point(foc, ref) if foc.size and ref.size else np.nan

    point = ps_weighted(w)
    if replicate_weights is None:
        return {"probability_superiority": point, "se": np.nan,
                "ci_low": np.nan, "ci_high": np.nan,
                "n_replicates": 0, "method": method}

    R = np.atleast_2d(np.asarray(replicate_weights, dtype=float))[:, ok]
    rep = np.array([ps_weighted(R[i]) for i in range(R.shape[0])])
    var = replicate_variance(point, rep, method, fay_factor, fpc)
    se = var["se"]
    crit = float(stats.norm.ppf(0.975))
    return {
        "probability_superiority": point,
        "se": se,
        "ci_low": float(np.clip(point - crit * se, 0, 1)) if np.isfinite(se) else np.nan,
        "ci_high": float(np.clip(point + crit * se, 0, 1)) if np.isfinite(se) else np.nan,
        "n_replicates": var["n_replicates"],
        "method": method,
    }


def ordinal_disparity(y_true, group, focal_label, n_levels=None):
    """Disparity measures for an ordered decision or rating.

    Binary fairness metrics collapse an ordered outcome to a single cut, which
    discards the information the ordering carries: a system that pushes the
    focal group from the top band to the second is treated identically to one
    that pushes them to the bottom. Three complementary measures are reported
    instead.

    ``mean_difference``
        Focal minus reference mean level, on the category scale.
    ``standardized_difference``
        The same difference divided by the pooled standard deviation, so it is
        comparable across scales of different width.
    ``probability_superiority``
        The probability that a randomly chosen focal case sits above a randomly
        chosen reference case, ties counted as half. Equal treatment gives 0.5;
        this is the ordinal analogue of a selection-rate ratio and, unlike the
        mean difference, does not assume the category spacing is meaningful.
    ``max_cumulative_gap``
        The largest gap in cumulative proportions across all thresholds, that
        is, the worst disparity any single cut point would reveal. This bounds
        what a binary analysis could have found had it chosen the least
        favourable threshold.

    Parameters
    ----------
    y_true : array-like
        Ordered category codes assigned by the system.
    group : array-like
        Binary group membership.
    focal_label : hashable
        Value of ``group`` identifying the focal group.
    n_levels : int, optional
        Number of categories. Inferred from the data when omitted.

    Returns
    -------
    dict

    Notes
    -----
    All measures are signed so that negative values, and a probability of
    superiority below 0.5, indicate the focal group receives lower outcomes.

    The probability of superiority carries a standard error, reported as
    ``ps_se`` with a Wald interval in ``ps_ci_low`` and ``ps_ci_high``. It uses
    the Hanley-McNeil approximation, which treats the two groups as
    independently sampled; under a clustered design it will be too small, and
    the estimate should be recomputed across replicate weights instead. Under
    complete separation, where the probability is exactly 0 or 1, that formula
    gives a variance of zero and hence a point interval; a conservative
    finite-sample bound is substituted so the interval still has width.
    """
    y = np.asarray(y_true).astype(float)
    g = np.asarray(pd.Series(group).to_numpy())
    ok = ~np.isnan(y)
    y, g = y[ok], g[ok]

    foc, ref = y[g == focal_label], y[g != focal_label]
    if foc.size == 0 or ref.size == 0:
        return {"metric": "ordinal_disparity", "mean_difference": np.nan,
                "standardized_difference": np.nan,
                "probability_superiority": np.nan, "ps_se": np.nan,
                "ps_ci_low": np.nan, "ps_ci_high": np.nan,
                "max_cumulative_gap": np.nan, "n_focal": int(foc.size),
                "n_reference": int(ref.size)}

    mean_diff = float(foc.mean() - ref.mean())
    pooled = np.sqrt(
        ((foc.size - 1) * foc.var(ddof=1) + (ref.size - 1) * ref.var(ddof=1))
        / max(foc.size + ref.size - 2, 1)
    ) if foc.size > 1 and ref.size > 1 else np.nan
    std_diff = float(mean_diff / pooled) if np.isfinite(pooled) and pooled > 0 else np.nan

    # Probability of superiority via the rank-sum identity, which avoids
    # forming the full n_focal x n_reference comparison matrix.
    ranks = stats.rankdata(np.concatenate([foc, ref]))
    r_foc = ranks[: foc.size].sum()
    m, n = foc.size, ref.size
    ps = float((r_foc - m * (m + 1) / 2) / (m * n))

    # Hanley-McNeil standard error of the probability of superiority.
    q1 = ps / (2 - ps) if ps < 1 else 1.0
    q2 = 2 * ps**2 / (1 + ps) if ps > 0 else 0.0
    var_ps = (
        ps * (1 - ps) + (m - 1) * (q1 - ps**2) + (n - 1) * (q2 - ps**2)
    ) / (m * n)
    if var_ps > 0:
        ps_se = float(np.sqrt(var_ps))
    elif ps in (0.0, 1.0):
        # Complete separation: every focal case falls below (or above) every
        # reference case. The Hanley-McNeil variance is zero there, which would
        # imply a point interval. Fall back on the conservative binomial-style
        # bound, which at least reflects that the sample is finite.
        ps_se = float(np.sqrt(1.0 / (4 * min(m, n))))
    else:
        ps_se = np.nan
    crit = float(stats.norm.ppf(0.975))
    ps_lo = float(np.clip(ps - crit * ps_se, 0.0, 1.0)) if np.isfinite(ps_se) else np.nan
    ps_hi = float(np.clip(ps + crit * ps_se, 0.0, 1.0)) if np.isfinite(ps_se) else np.nan

    levels = np.unique(y) if n_levels is None else np.arange(n_levels)
    gaps = [
        abs((foc <= lv).mean() - (ref <= lv).mean())
        for lv in levels[:-1]
    ]
    max_gap = float(max(gaps)) if gaps else np.nan

    return {
        "metric": "ordinal_disparity",
        "mean_difference": mean_diff,
        "standardized_difference": std_diff,
        "probability_superiority": ps,
        "ps_se": ps_se,
        "ps_ci_low": ps_lo,
        "ps_ci_high": ps_hi,
        "max_cumulative_gap": max_gap,
        "n_focal": int(foc.size),
        "n_reference": int(ref.size),
    }
