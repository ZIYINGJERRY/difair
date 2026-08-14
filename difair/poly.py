"""Differential item functioning for polytomous items.

Rating scales, partial-credit items and Likert questionnaires produce ordered
responses with more than two categories, which the dichotomous procedures in
:mod:`difair.dif` cannot accept. This module supplies the two procedures that
dominate operational practice for such items:

* :func:`generalized_mantel_haenszel` -- the Mantel test for ordered
  categorical data, stratified on the matching score, together with the
  standardized mean difference used as its effect size.
* :func:`ordinal_logistic_dif` -- the polytomous extension of the
  Swaminathan-Rogers procedure, fitting nested proportional-odds models and
  separating uniform from non-uniform DIF by likelihood-ratio tests.

Both accept ordered integer responses with any number of categories and return
tidy per-item results, matching the conventions of the dichotomous module: one
row per item, negative effect sizes indicating disadvantage to the focal group,
and an explicit ``focal_label`` wherever the group coding is not 0/1.

References
----------
Mantel, N. (1963). Chi-square tests with one degree of freedom: extensions of
    the Mantel-Haenszel procedure. *JASA, 58*, 690-700.
Zwick, R., & Thayer, D. T. (1996). Evaluating the magnitude of differential
    item functioning in polytomous items. *Journal of Educational and
    Behavioral Statistics, 21*, 187-201.
French, B. F., & Miller, F. G. (2016). Logistic regression and its use in
    detecting differential item functioning in polytomous items. *Journal of
    Educational Measurement, 53*, 42-60.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import expit

__all__ = [
    "generalized_mantel_haenszel",
    "ordinal_logistic_dif",
    "purify_poly_matching",
    "detect_dif_poly",
]


def _validate_poly(responses, group, focal_label):
    """Coerce inputs and check the ordered-categorical contract."""
    if isinstance(responses, pd.DataFrame):
        item_names = list(responses.columns)
        u = responses.to_numpy()
    else:
        u = np.asarray(responses)
        # Check the shape before deriving names from it: a one-dimensional
        # array would otherwise raise IndexError here rather than the
        # informative error below.
        if u.ndim != 2:
            raise ValueError(
                "`responses` must be 2-dimensional (n_persons x n_items); "
                f"got an array with {u.ndim} dimension(s)."
            )
        item_names = [f"item_{j + 1}" for j in range(u.shape[1])]

    u = u.astype(float)
    g = np.asarray(pd.Series(group).to_numpy())

    if u.ndim != 2:
        raise ValueError("`responses` must be 2-dimensional (n_persons x n_items).")
    if g.shape[0] != u.shape[0]:
        raise ValueError(
            f"`group` has {g.shape[0]} entries but `responses` has {u.shape[0]} rows."
        )

    finite = u[~np.isnan(u)]
    if finite.size and not np.allclose(finite, np.round(finite)):
        raise ValueError(
            "Polytomous responses must be integer category codes; "
            "non-integer values were found."
        )

    labels = pd.unique(pd.Series(g).dropna())
    if len(labels) != 2:
        raise ValueError(
            f"`group` must be binary (reference vs focal); found {len(labels)} levels."
        )
    if focal_label is None:
        as_set = set(labels)
        if as_set <= {0, 1} or as_set <= {True, False} or as_set <= {0.0, 1.0}:
            focal_label = max(labels, key=lambda v: float(v))
        else:
            raise ValueError(
                "`focal_label` must be given explicitly for non-binary labels. "
                f"`group` contains {sorted(map(str, labels))}. DIF statistics are "
                "directional, so an incorrect assignment reverses every sign."
            )
    elif focal_label not in labels:
        raise ValueError(f"focal_label={focal_label!r} not present in `group`.")

    constant = [
        item_names[j]
        for j in range(u.shape[1])
        if len(np.unique(u[~np.isnan(u[:, j]), j])) < 2
    ]
    if constant:
        warnings.warn(
            f"{len(constant)} item(s) have no variance: {constant[:5]}"
            f"{'...' if len(constant) > 5 else ''}. They yield undetermined "
            "statistics. Consider excluding them before analysis.",
            stacklevel=3,
        )

    return u, g == focal_label, item_names, focal_label


def _poly_matching_scores(u, matching, include_studied):
    """Matching totals for every item, computed once."""
    n, J = u.shape
    if matching is not None:
        col = np.asarray(matching, dtype=float)
    elif include_studied:
        col = np.nansum(u, axis=1)
    else:
        return np.nansum(u, axis=1)[:, None] - np.nan_to_num(u)
    return np.broadcast_to(col[:, None], (n, J))


def _n_categories(u):
    """Number of distinct response categories present, per item."""
    return [len(np.unique(col[~np.isnan(col)])) for col in u.T]


# --------------------------------------------------------------------------- #
# generalized Mantel-Haenszel
# --------------------------------------------------------------------------- #
def generalized_mantel_haenszel(
    responses,
    group,
    focal_label=None,
    matching=None,
    include_studied_item=True,
    weights=None,
    alpha=0.05,
):
    """Mantel test for ordered categorical items, with the SMD effect size.

    Within each level of the matching score the focal and reference groups are
    compared on the mean item score, using the category codes as scores. The
    statistic sums the stratum contributions and is asymptotically chi-square
    with one degree of freedom.

    The accompanying effect size is the standardized mean difference of Zwick
    and Thayer: the difference in weighted mean item scores divided by the
    total-group standard deviation. Negative values indicate that the focal
    group scores lower after matching, that is, DIF against the focal group.

    Parameters
    ----------
    responses : array-like or DataFrame, shape (n_persons, n_items)
        Ordered integer category codes. ``NaN`` marks a missing response.
    group : array-like, shape (n_persons,)
        Binary group membership.
    focal_label : hashable, optional
        Value of ``group`` identifying the focal group. Required unless the
        labels are 0/1 or boolean.
    matching : array-like, optional
        External matching criterion. Defaults to the observed total score.
    include_studied_item : bool, default True
        Whether the studied item contributes to the matching total.
    weights : array-like, optional
        Survey weights, one per respondent. Stratum totals become weighted
        totals, giving design-consistent point estimates. As in the dichotomous
        module the variance formula still assumes simple random sampling, so
        pair this with :func:`difair.survey.survey_dif` when design-based
        standard errors are needed.
    alpha : float, default 0.05
        Significance level for the ETS-style classification.

    Returns
    -------
    DataFrame
        Columns ``item``, ``n_categories``, ``chi2``, ``p_value``, ``smd``
        (standardized mean difference), ``smd_class``, ``n_strata``, ``favors``.

    Notes
    -----
    Classification follows the polytomous convention in operational use:
    ``|SMD| < 0.17`` negligible (A), ``0.17`` to ``0.25`` moderate (B),
    ``>= 0.25`` large (C), with significance required for B or C.
    """
    u, is_focal, item_names, _ = _validate_poly(responses, group, focal_label)
    scores = _poly_matching_scores(u, matching, include_studied_item)
    ncat = _n_categories(u)
    out = []

    for j, name in enumerate(item_names):
        y = u[:, j]
        s = scores[:, j]
        ok = ~np.isnan(y) & ~np.isnan(s)
        if weights is not None:
            ok &= ~np.isnan(np.asarray(weights, dtype=float))
        yj, sj, fj = y[ok], s[ok], is_focal[ok]
        wj = np.ones(yj.shape) if weights is None else np.asarray(weights, float)[ok]

        if yj.size < 2 or len(np.unique(yj)) < 2:
            out.append(_gmh_empty(name, ncat[j]))
            continue

        levels, idx = np.unique(sj, return_inverse=True)
        k = len(levels)
        cnt_all = np.bincount(idx, weights=wj, minlength=k)
        cnt_foc = np.bincount(idx[fj], weights=wj[fj], minlength=k)
        sum_all = np.bincount(idx, weights=wj * yj, minlength=k)
        sum_sq = np.bincount(idx, weights=wj * yj**2, minlength=k)
        sum_foc = np.bincount(idx[fj], weights=wj[fj] * yj[fj], minlength=k)

        keep = (cnt_foc > 0) & (cnt_foc < cnt_all)
        if not keep.any():
            out.append(_gmh_empty(name, ncat[j]))
            continue

        cnt_all, cnt_foc = cnt_all[keep], cnt_foc[keep]
        sum_all, sum_sq, sum_foc = sum_all[keep], sum_sq[keep], sum_foc[keep]

        # Mantel statistic: observed vs expected focal-group score total.
        expected = cnt_foc * sum_all / cnt_all
        with np.errstate(divide="ignore", invalid="ignore"):
            var = np.where(
                cnt_all > 1,
                cnt_foc * (cnt_all - cnt_foc)
                * (cnt_all * sum_sq - sum_all**2)
                / (cnt_all**2 * (cnt_all - 1)),
                0.0,
            )
        v = var.sum()
        chi2 = (sum_foc.sum() - expected.sum()) ** 2 / v if v > 0 else np.nan
        p = float(stats.chi2.sf(chi2, df=1)) if np.isfinite(chi2) else np.nan

        # Zwick-Thayer standardized mean difference. Weighting uses the total
        # stratum sizes, not the focal group's, so the statistic is exactly
        # antisymmetric under exchanging which group is focal; weighting by one
        # group's sizes would make the magnitude depend on that choice.
        w = cnt_all / cnt_all.sum()
        mean_foc = (w * (sum_foc / cnt_foc)).sum()
        mean_ref = (w * ((sum_all - sum_foc) / (cnt_all - cnt_foc))).sum()
        # Weighted standard deviation on the effective sample size, so that
        # integer weights reproduce the replicated-sample result exactly.
        sw = wj.sum()
        mu = np.average(yj, weights=wj)
        var = np.average((yj - mu) ** 2, weights=wj) * sw / max(sw - 1, 1)
        sd = np.sqrt(var)
        smd = float((mean_foc - mean_ref) / sd) if sd > 0 else np.nan

        out.append({
            "item": name,
            "n_categories": ncat[j],
            "chi2": float(chi2),
            "p_value": p,
            "smd": smd,
            "smd_class": _smd_class(smd, p, alpha),
            "n_strata": int(keep.sum()),
            "favors": ("focal" if smd > 0 else "reference" if smd < 0 else "neither")
            if np.isfinite(smd) else "undetermined",
        })

    return pd.DataFrame(out)


def _gmh_empty(name, ncat):
    return {
        "item": name, "n_categories": ncat, "chi2": np.nan, "p_value": np.nan,
        "smd": np.nan, "smd_class": "undetermined", "n_strata": 0,
        "favors": "undetermined",
    }


def _smd_class(smd, p, alpha):
    """ETS-style A/B/C bands for the standardized mean difference."""
    if not np.isfinite(smd) or not np.isfinite(p):
        return "undetermined"
    if p >= alpha or abs(smd) < 0.17:
        return "A"
    return "C" if abs(smd) >= 0.25 else "B"


# --------------------------------------------------------------------------- #
# ordinal logistic regression
# --------------------------------------------------------------------------- #
def ordinal_logistic_dif(
    responses,
    group,
    focal_label=None,
    matching=None,
    include_studied_item=True,
    standardize_matching=True,
):
    """Proportional-odds logistic regression DIF for ordered items.

    Fits three nested cumulative-logit models per item, where ``S`` is the
    matching score and ``G`` the focal-group indicator, and compares them by
    likelihood-ratio tests::

        M0:  logit P(Y <= k) = a_k - (b1*S)
        M1:  logit P(Y <= k) = a_k - (b1*S + b2*G)
        M2:  logit P(Y <= k) = a_k - (b1*S + b2*G + b3*S*G)

    Returns
    -------
    DataFrame
        Columns ``item``, ``n_categories``, ``chi2_total``/``p_total`` (2 df),
        ``chi2_uniform``/``p_uniform`` (1 df), ``chi2_nonuniform``/
        ``p_nonuniform`` (1 df), ``beta_group``, ``beta_interaction``.

    Notes
    -----
    Items whose model fails to converge, typically through quasi-complete
    separation or a sparsely populated category, yield ``NaN`` rather than an
    exception, so a single problematic item does not abort a test-level
    analysis.
    """
    u, is_focal, item_names, _ = _validate_poly(responses, group, focal_label)
    scores = _poly_matching_scores(u, matching, include_studied_item)
    ncat = _n_categories(u)
    g = is_focal.astype(float)
    out = []

    for j, name in enumerate(item_names):
        y = u[:, j]
        s = scores[:, j]
        ok = ~np.isnan(y) & ~np.isnan(s)
        yj, sj, gj = y[ok], s[ok].astype(float), g[ok]

        if yj.size < 30 or len(np.unique(yj)) < 2 or len(np.unique(gj)) < 2:
            out.append(_olr_empty(name, ncat[j]))
            continue

        if standardize_matching and sj.std() > 0:
            sj = (sj - sj.mean()) / sj.std()

        # Recode to consecutive ranks so no category is empty.
        _, yr = np.unique(yj, return_inverse=True)

        X0 = sj[:, None]
        X1 = np.column_stack([sj, gj])
        X2 = np.column_stack([sj, gj, sj * gj])

        n_cat = len(np.unique(yr))
        try:
            # Nested models: each solution warm-starts the next, with the new
            # coefficient initialised at zero.
            ll0, p0 = _fit_cumlogit(yr, X0, n_cat)
            ll1, p1 = _fit_cumlogit(yr, X1, n_cat, np.insert(p0, len(p0), 0.0))
            ll2, p2 = _fit_cumlogit(yr, X2, n_cat, np.insert(p1, len(p1), 0.0))
            if not all(np.isfinite([ll0, ll1, ll2])):
                raise ValueError("non-finite log-likelihood")
        except Exception:
            out.append(_olr_empty(name, ncat[j]))
            continue

        chi_tot = max(2 * (ll2 - ll0), 0.0)
        chi_uni = max(2 * (ll1 - ll0), 0.0)
        chi_non = max(2 * (ll2 - ll1), 0.0)

        out.append({
            "item": name,
            "n_categories": ncat[j],
            "chi2_total": float(chi_tot),
            "p_total": float(stats.chi2.sf(chi_tot, 2)),
            "chi2_uniform": float(chi_uni),
            "p_uniform": float(stats.chi2.sf(chi_uni, 1)),
            "chi2_nonuniform": float(chi_non),
            "p_nonuniform": float(stats.chi2.sf(chi_non, 1)),
            "beta_group": float(p2[-2]),
            "beta_interaction": float(p2[-1]),
        })

    return pd.DataFrame(out)


def _cumlogit_nll(params, y, X, n_cat):
    """Negative log-likelihood of a proportional-odds model.

    Parameterised by ``n_cat - 1`` cutpoints, held ordered through cumulative
    softplus increments, followed by the regression coefficients. Implemented
    directly rather than through a generic distribution interface: the
    cumulative logit is just the logistic sigmoid, and calling it through
    ``scipy.stats``'s dispatch machinery dominated the run time.
    """
    k = n_cat - 1
    first, incr, beta = params[0], params[1:k], params[k:]
    cuts = np.concatenate([[first], first + np.cumsum(np.exp(incr))]) if k > 1 \
        else np.array([first])

    eta = X @ beta if beta.size else np.zeros(len(y))
    z = cuts[None, :] - eta[:, None]
    cum = expit(z)                                   # P(Y <= k)
    cum = np.concatenate([np.zeros((len(y), 1)), cum, np.ones((len(y), 1))], axis=1)
    prob = np.clip(np.diff(cum, axis=1)[np.arange(len(y)), y], 1e-12, None)
    return -np.log(prob).sum()


def _fit_cumlogit(y, X, n_cat, start=None):
    """Fit a proportional-odds model, returning (log-likelihood, coefficients)."""
    k = n_cat - 1
    n_beta = X.shape[1] if X.ndim > 1 else 0
    if start is None:
        # Cutpoints at the empirical cumulative logits, coefficients at zero.
        freq = np.bincount(y, minlength=n_cat) / len(y)
        cum = np.clip(np.cumsum(freq)[:-1], 1e-3, 1 - 1e-3)
        q = np.log(cum / (1 - cum))
        start = np.concatenate([
            [q[0]],
            np.log(np.clip(np.diff(q), 1e-3, None)) if k > 1 else [],
            np.zeros(n_beta),
        ])
    res = optimize.minimize(
        _cumlogit_nll, start, args=(y, X, n_cat),
        method="BFGS", options={"maxiter": 300, "gtol": 1e-6},
    )
    return -float(res.fun), res.x


def _olr_empty(name, ncat):
    return {
        "item": name, "n_categories": ncat,
        "chi2_total": np.nan, "p_total": np.nan,
        "chi2_uniform": np.nan, "p_uniform": np.nan,
        "chi2_nonuniform": np.nan, "p_nonuniform": np.nan,
        "beta_group": np.nan, "beta_interaction": np.nan,
    }


def purify_poly_matching(
    responses, group, focal_label=None, max_iter=5, alpha=0.05
):
    """Iteratively purify the matching total for polytomous items.

    The matching criterion is built from the items under test, so an item
    carrying DIF biases the yardstick applied to every other item. Each pass
    recomputes the total from items not currently flagged at SMD class B or C,
    stopping when the flagged set is stable.

    Returns
    -------
    (matching, flagged) : (ndarray, list of str)

    Notes
    -----
    Purification assumes the flagged items are a minority: it rebuilds the
    criterion from the items believed clean, which requires enough of them to
    remain. In simulation the routine recovers the planted set exactly with no
    false positives while up to roughly a quarter of items carry DIF, but at
    half it breaks down and flags the entire instrument, because by then the
    criterion is contaminated whichever items are dropped. A result flagging
    nearly every item should be read as a failure of the matching assumption
    rather than as evidence about the items.
    """
    u, is_focal, item_names, _ = _validate_poly(responses, group, focal_label)
    flagged: set = set()

    for _ in range(max_iter):
        keep = [i for i, nm in enumerate(item_names) if nm not in flagged]
        if len(keep) < 2:
            warnings.warn("Purification removed nearly every item; stopping early.")
            break
        matching = np.nansum(u[:, keep], axis=1)
        res = generalized_mantel_haenszel(
            pd.DataFrame(u, columns=item_names), is_focal, focal_label=True,
            matching=matching, alpha=alpha,
        )
        new = set(res.loc[res.smd_class.isin(["B", "C"]), "item"])
        if new == flagged:
            break
        flagged = new

    keep = [i for i, nm in enumerate(item_names) if nm not in flagged]
    matching = np.nansum(u[:, keep], axis=1) if keep else np.nansum(u, axis=1)
    return matching, sorted(flagged)


# --------------------------------------------------------------------------- #
def detect_dif_poly(
    responses,
    group,
    focal_label=None,
    methods=("gmh", "ordinal"),
    purify=False,
    weights=None,
    alpha=0.05,
):
    """Run the polytomous procedures and merge them into one table per item.

    Parameters
    ----------
    methods : tuple of str
        Any of ``"gmh"`` (generalized Mantel-Haenszel) and ``"ordinal"``
        (proportional-odds logistic regression).
    purify : bool, default False
        Purify the matching total before testing, as the dichotomous
        :func:`difair.dif.detect_dif` does.

    Returns
    -------
    DIFResult
        Same container as the dichotomous :func:`difair.dif.detect_dif`, so
        downstream code and :func:`difair.report.audit_report` accept either.
        ``.flagged`` lists items reaching SMD class B or C.
    """
    from difair.dif import DIFResult

    matching = None
    if purify:
        matching, _ = purify_poly_matching(responses, group, focal_label, alpha=alpha)

    frames = []
    if "gmh" in methods:
        frames.append(generalized_mantel_haenszel(
            responses, group, focal_label, matching=matching,
            weights=weights, alpha=alpha))
    if "ordinal" in methods:
        frames.append(ordinal_logistic_dif(
            responses, group, focal_label, matching=matching))
    if not frames:
        raise ValueError("`methods` selected no procedure.")

    table = frames[0]
    for extra in frames[1:]:
        dup = [c for c in extra.columns if c in table.columns and c != "item"]
        table = table.merge(extra.drop(columns=dup), on="item", how="outer")

    flagged = (
        sorted(table.loc[table.smd_class.isin(["B", "C"]), "item"])
        if "smd_class" in table else []
    )
    _, _, _, focal = _validate_poly(responses, group, focal_label)
    return DIFResult(table=table, flagged=flagged, focal_label=focal, purified=purify)
