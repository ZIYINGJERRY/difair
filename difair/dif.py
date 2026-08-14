"""Item-level differential item functioning (DIF) detection.

Implements the four DIF families that dominate operational testing practice:

* ``mantel_haenszel`` -- Mantel-Haenszel common odds ratio, chi-square with
  continuity correction, Robins-Breslow-Greenland standard error, and the ETS
  delta scale with A/B/C classification.
* ``logistic_dif`` -- Swaminathan & Rogers logistic regression, separating
  uniform from non-uniform DIF via nested likelihood-ratio tests, with the
  Zumbo-Thomas Nagelkerke Delta-R-squared effect size.
* ``standardization`` -- Dorans & Kulick standardized proportion difference.
* ``breslow_day`` -- Breslow-Day (optionally Tarone-corrected) test of odds
  ratio homogeneity, a non-uniform DIF indicator.

All functions accept a matrix of dichotomous item responses and a binary group
vector, and return tidy :class:`pandas.DataFrame` results with one row per item.

References
----------
Holland, P. W., & Thayer, D. T. (1988). Differential item performance and the
    Mantel-Haenszel procedure.
Swaminathan, H., & Rogers, H. J. (1990). Detecting differential item
    functioning using logistic regression procedures. *Journal of Educational
    Measurement, 27*, 361-370.
Dorans, N. J., & Kulick, E. (1986). Demonstrating the utility of the
    standardization approach. *Journal of Educational Measurement, 23*, 355-368.
Breslow, N. E., & Day, N. E. (1980). *Statistical Methods in Cancer Research.*
Zumbo, B. D. (1999). *A Handbook on the Theory and Methods of DIF.*
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "mantel_haenszel",
    "logistic_dif",
    "standardization",
    "breslow_day",
    "detect_dif",
    "purify_matching_score",
    "DIFResult",
    "normalize_weights",
]

_ETS_DELTA_SCALE = -2.35


# --------------------------------------------------------------------------- #
# input validation
# --------------------------------------------------------------------------- #
def _validate(responses, group, focal_label):
    """Coerce inputs to arrays and check the dichotomous / binary contract."""
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

    observed = np.unique(u[~np.isnan(u)])
    if not np.all(np.isin(observed, [0.0, 1.0])):
        raise ValueError(
            "Only dichotomous (0/1) responses are supported in this release; "
            f"found values {observed[:6]}."
        )

    # Items answered identically by everyone carry no information and, because
    # they enter the matching total as a constant, can collapse the stratum
    # structure that every procedure here depends on. Warn rather than drop:
    # which items to exclude is the analyst's decision.
    with np.errstate(invalid="ignore"):
        constant = [
            item_names[j]
            for j in range(u.shape[1])
            if len(np.unique(u[~np.isnan(u[:, j]), j])) < 2
        ]
    if constant:
        warnings.warn(
            f"{len(constant)} item(s) have no variance and were answered "
            f"identically by every respondent: {constant[:5]}"
            f"{'...' if len(constant) > 5 else ''}. They yield undetermined "
            "statistics and inflate the matching score by a constant. Consider "
            "excluding them before analysis.",
            stacklevel=3,
        )

    labels = pd.unique(pd.Series(g).dropna())
    if len(labels) != 2:
        raise ValueError(
            f"`group` must be binary (reference vs focal); found {len(labels)} levels."
        )

    if focal_label is None:
        # Only boolean / 0-1 indicators carry an unambiguous convention
        # (1 or True denotes the focal group). For any other labelling the
        # caller must say which group is focal: silently guessing would flip
        # the sign of every statistic in this module.
        as_set = set(labels)
        if as_set <= {0, 1} or as_set <= {True, False} or as_set <= {0.0, 1.0}:
            focal_label = max(labels, key=lambda v: float(v))
        else:
            raise ValueError(
                "`focal_label` must be given explicitly for non-binary labels. "
                f"`group` contains {sorted(map(str, labels))}; pass e.g. "
                f"focal_label={sorted(map(str, labels))[0]!r} to name the focal "
                "group. DIF statistics are directional, so an incorrect "
                "assignment silently reverses every sign."
            )
    elif focal_label not in labels:
        raise ValueError(f"focal_label={focal_label!r} not present in `group`.")

    is_focal = g == focal_label
    return u, is_focal, item_names, focal_label


def _matching_score(u, matching, item_index, include_studied):
    """Total-score matching criterion, optionally leaving the studied item out."""
    if matching is not None:
        return np.asarray(matching, dtype=float)
    if include_studied:
        return np.nansum(u, axis=1)
    keep = np.ones(u.shape[1], dtype=bool)
    keep[item_index] = False
    return np.nansum(u[:, keep], axis=1)


def _matching_scores_all(u, matching, include_studied):
    """Matching scores for every item at once.

    The rest-score variant is the row total minus the studied item, so the
    whole matrix follows from one ``nansum`` instead of one per item. Returns
    an ``(n_persons, n_items)`` array whose column ``j`` is the criterion for
    item ``j``; when an external criterion is supplied, or the studied item is
    included, every column is identical and the array is a broadcast view.
    """
    n, J = u.shape
    if matching is not None:
        col = np.asarray(matching, dtype=float)
    elif include_studied:
        col = np.nansum(u, axis=1)
    else:
        total = np.nansum(u, axis=1)
        return total[:, None] - np.nan_to_num(u)
    return np.broadcast_to(col[:, None], (n, J))


def _strata_tables(u_item, is_focal, score, weights=None):
    """Build per-stratum 2x2 tables, dropping strata with an empty group.

    Counts are accumulated with ``bincount`` over the integer-coded strata
    rather than by looping, so cost is linear in the number of respondents and
    independent of the number of distinct score levels.

    When ``weights`` is supplied the cells hold weighted totals rather than raw
    counts, which is what survey estimation requires: a respondent selected
    with probability ``p`` stands for ``1/p`` members of the population.
    """
    ok = ~np.isnan(u_item) & ~np.isnan(score)
    if weights is not None:
        ok &= ~np.isnan(weights)
    u_item, is_focal, score = u_item[ok], is_focal[ok], score[ok]
    w = np.ones(u_item.shape) if weights is None else np.asarray(weights, float)[ok]

    cols = ["stratum", "a", "b", "c", "d", "n_r", "n_f", "m1", "m0", "n_tot"]
    if u_item.size == 0:
        return pd.DataFrame(columns=cols)

    levels, idx = np.unique(score, return_inverse=True)
    k = len(levels)
    foc = is_focal
    ref = ~foc
    correct = u_item == 1

    n_r = np.bincount(idx[ref], weights=w[ref], minlength=k)
    n_f = np.bincount(idx[foc], weights=w[foc], minlength=k)
    a = np.bincount(idx[ref & correct], weights=w[ref & correct], minlength=k)
    c = np.bincount(idx[foc & correct], weights=w[foc & correct], minlength=k)

    keep = (n_r > 0) & (n_f > 0)  # strata carrying between-group information
    if not keep.any():
        return pd.DataFrame(columns=cols)

    levels, n_r, n_f, a, c = levels[keep], n_r[keep], n_f[keep], a[keep], c[keep]
    b, d = n_r - a, n_f - c
    return pd.DataFrame({
        "stratum": levels, "a": a, "b": b, "c": c, "d": d,
        "n_r": n_r, "n_f": n_f, "m1": a + c, "m0": b + d, "n_tot": n_r + n_f,
    })


# --------------------------------------------------------------------------- #
# Mantel-Haenszel
# --------------------------------------------------------------------------- #
def mantel_haenszel(
    responses,
    group,
    focal_label=None,
    matching=None,
    include_studied_item=True,
    correct=True,
    clamp_correction=True,
    weights=None,
    alpha=0.05,
):
    """Mantel-Haenszel DIF statistic with the ETS delta scale.

    Parameters
    ----------
    responses : array-like or DataFrame, shape (n_persons, n_items)
        Dichotomous (0/1) item responses. ``NaN`` marks a missing response.
    group : array-like, shape (n_persons,)
        Binary group membership.
    focal_label : hashable, optional
        Value of ``group`` identifying the focal group. Defaults to the larger
        label under string ordering.
    matching : array-like, optional
        External matching criterion. Defaults to the observed total score.
    include_studied_item : bool, default True
        Whether the studied item contributes to the matching score. The
        operational ETS convention includes it; set ``False`` for the
        rest-score variant.
    correct : bool, default True
        Apply the 0.5 continuity correction to the chi-square statistic.
    clamp_correction : bool, default True
        Floor the corrected numerator at zero. Without it, ``(|d| - 0.5)**2``
        grows again as the raw difference ``d`` falls below 0.5, so an item with
        no group difference at all receives a non-zero statistic. Reference
        implementations (``difR``, ``stats::mantelhaen.test``) do not clamp;
        set this to ``False`` to reproduce them exactly. The two differ only
        for items whose chi-square is already far below any critical value.
    weights : array-like, optional
        Survey weights, one per respondent. Cells become weighted totals, which
        is what design-based estimation of a population quantity requires.
        Point estimates are then consistent for the population; the variance
        formulas here still assume simple random sampling, so standard errors
        and p-values are anti-conservative under a clustered design and should
        be treated as indicative. See :func:`normalize_weights`.
    alpha : float, default 0.05
        Significance level used by the ETS A/B/C classification.

    Returns
    -------
    DataFrame
        One row per item with columns ``item``, ``alpha_mh`` (common odds
        ratio), ``delta_mh`` (ETS delta), ``se_log_alpha``, ``chi2``,
        ``p_value``, ``ets_class``, ``n_strata``, ``favors``.

    Notes
    -----
    ``alpha_mh`` is the reference-to-focal odds ratio, so ``alpha_mh > 1``
    means the item is easier for the reference group. The ETS delta then turns
    negative, matching the convention that negative delta signals DIF against
    the focal group.
    """
    u, is_focal, item_names, _ = _validate(responses, group, focal_label)
    scores = _matching_scores_all(u, matching, include_studied_item)
    out = []

    for j, name in enumerate(item_names):
        tab = _strata_tables(u[:, j], is_focal, scores[:, j], weights)

        if tab.empty:
            out.append(_mh_empty(name))
            continue

        a, b, c, d = tab["a"].values, tab["b"].values, tab["c"].values, tab["d"].values
        n_r, n_f = tab["n_r"].values, tab["n_f"].values
        m1, m0, T = tab["m1"].values, tab["m0"].values, tab["n_tot"].values

        R = a * d / T
        S = b * c / T
        sum_R, sum_S = R.sum(), S.sum()

        if sum_S <= 0 or sum_R <= 0:
            # Degenerate: the odds ratio is 0 or infinite.
            out.append(_mh_empty(name, n_strata=len(tab)))
            continue

        alpha_mh = sum_R / sum_S
        delta = _ETS_DELTA_SCALE * np.log(alpha_mh)

        # chi-square with continuity correction
        E = n_r * m1 / T
        with np.errstate(divide="ignore", invalid="ignore"):
            V = np.where(T > 1, n_r * n_f * m1 * m0 / (T**2 * (T - 1)), 0.0)
        num = abs(a.sum() - E.sum()) - (0.5 if correct else 0.0)
        if clamp_correction:
            num = max(num, 0.0)
        chi2 = num**2 / V.sum() if V.sum() > 0 else np.nan
        p = float(stats.chi2.sf(chi2, df=1)) if np.isfinite(chi2) else np.nan

        # Robins-Breslow-Greenland variance of log(alpha_mh)
        P = (a + d) / T
        Q = (b + c) / T
        var_log = (
            (P * R).sum() / (2 * sum_R**2)
            + (P * S + Q * R).sum() / (2 * sum_R * sum_S)
            + (Q * S).sum() / (2 * sum_S**2)
        )
        se_log = float(np.sqrt(var_log)) if var_log > 0 else np.nan

        out.append(
            {
                "item": name,
                "alpha_mh": float(alpha_mh),
                "delta_mh": float(delta),
                "se_log_alpha": se_log,
                "chi2": float(chi2),
                "p_value": p,
                "ets_class": _ets_class(delta, p, alpha),
                "n_strata": int(len(tab)),
                "favors": _favors(delta),
            }
        )

    return pd.DataFrame(out)


def _mh_empty(name, n_strata=0):
    return {
        "item": name,
        "alpha_mh": np.nan,
        "delta_mh": np.nan,
        "se_log_alpha": np.nan,
        "chi2": np.nan,
        "p_value": np.nan,
        "ets_class": "undetermined",
        "n_strata": n_strata,
        "favors": "undetermined",
    }


def _ets_class(delta, p, alpha):
    """ETS A/B/C classification of DIF magnitude."""
    if not np.isfinite(delta) or not np.isfinite(p):
        return "undetermined"
    if p >= alpha or abs(delta) < 1.0:
        return "A"
    return "C" if abs(delta) >= 1.5 else "B"


def _favors(delta):
    if not np.isfinite(delta):
        return "undetermined"
    if delta < 0:
        return "reference"
    if delta > 0:
        return "focal"
    return "neither"


# --------------------------------------------------------------------------- #
# logistic regression DIF
# --------------------------------------------------------------------------- #
def logistic_dif(
    responses,
    group,
    focal_label=None,
    matching=None,
    include_studied_item=True,
    standardize_matching=True,
):
    """Swaminathan-Rogers logistic regression DIF.

    Fits three nested models per item, where ``S`` is the matching score and
    ``G`` the focal-group indicator::

        M0:  logit(p) = b0 + b1*S
        M1:  logit(p) = b0 + b1*S + b2*G
        M2:  logit(p) = b0 + b1*S + b2*G + b3*S*G

    and reports likelihood-ratio tests for uniform DIF (M1 vs M0), non-uniform
    DIF (M2 vs M1), and their joint effect (M2 vs M0).

    Returns
    -------
    DataFrame
        Columns ``item``, ``chi2_total``/``p_total`` (2 df), ``chi2_uniform``/
        ``p_uniform`` (1 df), ``chi2_nonuniform``/``p_nonuniform`` (1 df),
        ``beta_group``, ``beta_interaction``, ``delta_r2`` (Zumbo-Thomas
        Nagelkerke effect size) and ``zt_class``.
    """
    import statsmodels.api as sm

    u, is_focal, item_names, _ = _validate(responses, group, focal_label)
    scores = _matching_scores_all(u, matching, include_studied_item)
    g = is_focal.astype(float)
    out = []

    for j, name in enumerate(item_names):
        y = u[:, j]
        score = scores[:, j]
        ok = ~np.isnan(y) & ~np.isnan(score)
        yj, sj, gj = y[ok], score[ok].astype(float), g[ok]

        if yj.size < 20 or len(np.unique(yj)) < 2 or len(np.unique(gj)) < 2:
            out.append(_logistic_empty(name))
            continue

        if standardize_matching and sj.std() > 0:
            sj = (sj - sj.mean()) / sj.std()

        X0 = sm.add_constant(sj[:, None], has_constant="add")
        X1 = sm.add_constant(np.column_stack([sj, gj]), has_constant="add")
        X2 = sm.add_constant(np.column_stack([sj, gj, sj * gj]), has_constant="add")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Newton-Raphson with a tight tolerance: likelihood-ratio
                # statistics are differences of log-likelihoods, so loose
                # convergence shows up directly in the test statistic.
                fit = dict(disp=0, method="newton", maxiter=200, tol=1e-10)
                m0 = sm.Logit(yj, X0).fit(**fit)
                m1 = sm.Logit(yj, X1).fit(**fit)
                m2 = sm.Logit(yj, X2).fit(**fit)
        except Exception:  # separation, singular design, non-convergence
            out.append(_logistic_empty(name))
            continue

        chi_tot = max(2 * (m2.llf - m0.llf), 0.0)
        chi_uni = max(2 * (m1.llf - m0.llf), 0.0)
        chi_non = max(2 * (m2.llf - m1.llf), 0.0)
        dr2 = max(_nagelkerke(m2, yj) - _nagelkerke(m0, yj), 0.0)

        out.append(
            {
                "item": name,
                "chi2_total": float(chi_tot),
                "p_total": float(stats.chi2.sf(chi_tot, 2)),
                "chi2_uniform": float(chi_uni),
                "p_uniform": float(stats.chi2.sf(chi_uni, 1)),
                "chi2_nonuniform": float(chi_non),
                "p_nonuniform": float(stats.chi2.sf(chi_non, 1)),
                "beta_group": float(m2.params[2]),
                "beta_interaction": float(m2.params[3]),
                "delta_r2": float(dr2),
                "zt_class": _zumbo_thomas_class(dr2),
            }
        )

    return pd.DataFrame(out)


def _logistic_empty(name):
    return {
        "item": name,
        "chi2_total": np.nan,
        "p_total": np.nan,
        "chi2_uniform": np.nan,
        "p_uniform": np.nan,
        "chi2_nonuniform": np.nan,
        "p_nonuniform": np.nan,
        "beta_group": np.nan,
        "beta_interaction": np.nan,
        "delta_r2": np.nan,
        "zt_class": "undetermined",
    }


def _nagelkerke(model, y):
    """Nagelkerke pseudo R-squared."""
    n = len(y)
    p = y.mean()
    ll_null = n * (p * np.log(p) + (1 - p) * np.log(1 - p)) if 0 < p < 1 else 0.0
    if ll_null == 0:
        return 0.0
    cox_snell = 1 - np.exp((2 / n) * (ll_null - model.llf))
    denom = 1 - np.exp((2 / n) * ll_null)
    return float(cox_snell / denom) if denom > 0 else 0.0


def _zumbo_thomas_class(dr2):
    """Jodoin & Gierl (2001) effect-size bands for Zumbo-Thomas Delta-R2."""
    if not np.isfinite(dr2):
        return "undetermined"
    if dr2 < 0.035:
        return "negligible"
    return "large" if dr2 >= 0.070 else "moderate"


# --------------------------------------------------------------------------- #
# standardization
# --------------------------------------------------------------------------- #
def standardization(
    responses,
    group,
    focal_label=None,
    matching=None,
    include_studied_item=True,
    weights=None,
):
    """Dorans-Kulick standardized proportion difference (STD P-DIF).

    Weights the within-stratum proportion-correct difference by the focal
    group's stratum sizes. Negative values indicate DIF against the focal
    group. Conventional bands: ``|STD P-DIF| < 0.05`` negligible,
    ``0.05-0.10`` moderate, ``>= 0.10`` large.
    """
    u, is_focal, item_names, _ = _validate(responses, group, focal_label)
    scores = _matching_scores_all(u, matching, include_studied_item)
    out = []

    for j, name in enumerate(item_names):
        tab = _strata_tables(u[:, j], is_focal, scores[:, j], weights)
        if tab.empty:
            out.append({"item": name, "std_p_dif": np.nan, "std_class": "undetermined",
                        "favors": "undetermined", "n_strata": 0})
            continue

        p_f = tab["c"].values / tab["n_f"].values
        p_r = tab["a"].values / tab["n_r"].values
        w = tab["n_f"].values.astype(float)
        std = float((w * (p_f - p_r)).sum() / w.sum())

        out.append(
            {
                "item": name,
                "std_p_dif": std,
                "std_class": _std_class(std),
                "favors": "focal" if std > 0 else ("reference" if std < 0 else "neither"),
                "n_strata": int(len(tab)),
            }
        )

    return pd.DataFrame(out)


def _std_class(std):
    if not np.isfinite(std):
        return "undetermined"
    a = abs(std)
    if a < 0.05:
        return "negligible"
    return "large" if a >= 0.10 else "moderate"


# --------------------------------------------------------------------------- #
# Breslow-Day
# --------------------------------------------------------------------------- #
def breslow_day(
    responses,
    group,
    focal_label=None,
    matching=None,
    include_studied_item=True,
    tarone=True,
    weights=None,
):
    """Breslow-Day test of odds ratio homogeneity across strata.

    A significant result indicates that the reference-focal odds ratio varies
    with the matching score, i.e. non-uniform DIF. With ``tarone=True`` the
    Tarone correction is applied so the statistic is asymptotically chi-square
    with ``n_strata - 1`` degrees of freedom.
    """
    u, is_focal, item_names, _ = _validate(responses, group, focal_label)
    scores = _matching_scores_all(u, matching, include_studied_item)
    out = []

    for j, name in enumerate(item_names):
        tab = _strata_tables(u[:, j], is_focal, scores[:, j], weights)
        tab = tab[tab["n_tot"] > 1]

        if len(tab) < 2:
            out.append({"item": name, "bd_stat": np.nan, "df": 0, "p_value": np.nan,
                        "nonuniform_flag": False})
            continue

        a, n_r, m1, T = tab["a"].values, tab["n_r"].values, tab["m1"].values, tab["n_tot"].values
        R = tab["a"].values * tab["d"].values / T
        S = tab["b"].values * tab["c"].values / T
        if S.sum() <= 0 or R.sum() <= 0:
            out.append({"item": name, "bd_stat": np.nan, "df": 0, "p_value": np.nan,
                        "nonuniform_flag": False})
            continue
        psi = R.sum() / S.sum()

        stat, var_sum, e_sum = 0.0, 0.0, 0.0
        for ak, nrk, m1k, Tk in zip(a, n_r, m1, T):
            e = _bd_expected(psi, nrk, m1k, Tk)
            if e is None:
                continue
            b_, c_, d_ = nrk - e, m1k - e, Tk - nrk - m1k + e
            if min(e, b_, c_, d_) <= 0:
                continue
            var = 1.0 / (1.0 / e + 1.0 / b_ + 1.0 / c_ + 1.0 / d_)
            stat += (ak - e) ** 2 / var
            var_sum += var
            e_sum += e

        df = int(len(tab) - 1)
        if tarone and var_sum > 0:
            stat -= (a.sum() - e_sum) ** 2 / var_sum
        stat = max(stat, 0.0)
        p = float(stats.chi2.sf(stat, df)) if df > 0 else np.nan

        out.append(
            {
                "item": name,
                "bd_stat": float(stat),
                "df": df,
                "p_value": p,
                "nonuniform_flag": bool(np.isfinite(p) and p < 0.05),
            }
        )

    return pd.DataFrame(out)


def _bd_expected(psi, n_r, m1, T):
    """Expected reference-correct count under a common odds ratio ``psi``.

    Solves ``(1-psi)a^2 + [(T-n_r-m1) + psi(n_r+m1)]a - psi*n_r*m1 = 0`` and
    returns the root inside the feasible range for the 2x2 margins.
    """
    lo = max(0.0, n_r + m1 - T)
    hi = min(float(n_r), float(m1))
    if hi <= lo:
        return None

    A = 1.0 - psi
    B = (T - n_r - m1) + psi * (n_r + m1)
    C = -psi * n_r * m1

    if abs(A) < 1e-12:  # psi == 1 -> linear
        if abs(B) < 1e-12:
            return None
        root = -C / B
        return root if lo < root < hi else None

    disc = B * B - 4 * A * C
    if disc < 0:
        return None
    sq = np.sqrt(disc)
    for root in ((-B + sq) / (2 * A), (-B - sq) / (2 * A)):
        if lo < root < hi:
            return float(root)
    return None


# --------------------------------------------------------------------------- #
# purification and unified entry point
# --------------------------------------------------------------------------- #
def purify_matching_score(
    responses, group, focal_label=None, max_iter=5, alpha=0.05, verbose=False
):
    """Iteratively purify the matching score by removing DIF-flagged items.

    At each pass the matching total is recomputed from items not currently
    flagged at ETS level B or C. Iteration stops when the flagged set is
    stable or ``max_iter`` is reached.

    Returns
    -------
    (matching, flagged) : (ndarray, list of str)
    """
    u, is_focal, item_names, focal = _validate(responses, group, focal_label)
    flagged: set[str] = set()

    for it in range(max_iter):
        keep = [i for i, nm in enumerate(item_names) if nm not in flagged]
        if len(keep) < 2:
            warnings.warn("Purification removed nearly every item; stopping early.")
            break
        matching = np.nansum(u[:, keep], axis=1)
        res = mantel_haenszel(
            pd.DataFrame(u, columns=item_names), is_focal, focal_label=True,
            matching=matching, alpha=alpha,
        )
        new = set(res.loc[res.ets_class.isin(["B", "C"]), "item"])
        if verbose:
            print(f"[purify] pass {it + 1}: {len(new)} flagged")
        if new == flagged:
            break
        flagged = new

    keep = [i for i, nm in enumerate(item_names) if nm not in flagged]
    matching = np.nansum(u[:, keep], axis=1) if keep else np.nansum(u, axis=1)
    return matching, sorted(flagged)


@dataclass
class DIFResult:
    """Container returned by :func:`detect_dif`."""

    table: pd.DataFrame
    flagged: list
    focal_label: object
    purified: bool

    def __repr__(self):  # pragma: no cover - display helper
        return (
            f"<DIFResult items={len(self.table)} flagged={len(self.flagged)} "
            f"focal={self.focal_label!r} purified={self.purified}>"
        )

    def summary(self):
        """Counts of items in each ETS class."""
        return self.table.ets_class.value_counts().rename("n_items").to_frame()


def detect_dif(
    responses,
    group,
    focal_label=None,
    methods=("mh", "logistic", "std", "bd"),
    purify=False,
    alpha=0.05,
    **mh_kwargs,
):
    """Run several DIF procedures and merge them into one table per item.

    Parameters
    ----------
    methods : tuple of str
        Any of ``"mh"``, ``"logistic"``, ``"std"``, ``"bd"``.
    purify : bool, default False
        Purify the matching score before testing (see
        :func:`purify_matching_score`).

    Returns
    -------
    DIFResult
        ``.table`` holds the merged per-item statistics; ``.flagged`` lists
        items reaching ETS class B or C.
    """
    matching = None
    if purify:
        matching, _ = purify_matching_score(responses, group, focal_label, alpha=alpha)

    frames = []
    if "mh" in methods:
        frames.append(mantel_haenszel(responses, group, focal_label, matching, alpha=alpha, **mh_kwargs))
    if "logistic" in methods:
        frames.append(logistic_dif(responses, group, focal_label, matching))
    if "std" in methods:
        frames.append(standardization(responses, group, focal_label, matching))
    if "bd" in methods:
        frames.append(breslow_day(responses, group, focal_label, matching))

    if not frames:
        raise ValueError("`methods` selected no procedure.")

    table = frames[0]
    for extra in frames[1:]:
        dup = [c for c in extra.columns if c in table.columns and c != "item"]
        table = table.merge(extra.drop(columns=dup), on="item", how="outer")

    flagged = (
        sorted(table.loc[table.ets_class.isin(["B", "C"]), "item"])
        if "ets_class" in table
        else []
    )
    _, _, _, focal = _validate(responses, group, focal_label)
    return DIFResult(table=table, flagged=flagged, focal_label=focal, purified=purify)


def normalize_weights(weights, group=None):
    """Scale survey weights to sum to the sample size.

    Large-scale assessments distribute weights on the population scale, where a
    single respondent may stand for thousands of students. Chi-square
    statistics computed from such totals are inflated by the scale factor
    alone and are not interpretable. Rescaling so the weights sum to the number
    of respondents preserves the relative weighting, and therefore the point
    estimates, while returning the test statistics to their nominal scale.

    Parameters
    ----------
    weights : array-like
        Raw survey weights.
    group : array-like, optional
        If supplied, weights are normalised within each group, so each group
        contributes its own sample size. Use this when group sizes in the
        weighted population differ markedly from the sample.

    Returns
    -------
    ndarray

    Notes
    -----
    Normalisation does not make the variance estimates design-consistent. For
    published standard errors under a clustered design, replicate weights or a
    survey-specific package remain necessary; DIFair's weighted statistics are
    point estimates with indicative inference.
    """
    w = np.asarray(weights, dtype=float)
    if np.any(w[~np.isnan(w)] < 0):
        raise ValueError("Survey weights must be non-negative.")

    if group is None:
        total = np.nansum(w)
        if total <= 0:
            raise ValueError("Survey weights sum to zero.")
        return w * (np.isfinite(w).sum() / total)

    g = np.asarray(pd.Series(group).to_numpy())
    out = w.astype(float).copy()
    for lab in pd.unique(pd.Series(g).dropna()):
        m = (g == lab) & np.isfinite(w)
        tot = w[m].sum()
        if tot > 0:
            out[m] = w[m] * (m.sum() / tot)
    return out
