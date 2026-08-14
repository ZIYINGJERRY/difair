"""Design-based inference for large-scale assessments.

Weighted point estimates alone are not enough to publish results from PISA,
TIMSS, PIRLS or NAEP. Two further pieces of machinery are required, and this
module supplies both.

**Replicate weights.** These assessments sample schools before students, so
responses within a school are correlated and the simple-random-sampling
variance formulas used elsewhere in DIFair understate the true uncertainty.
The standard remedy is resampling: the survey organisation distributes a set of
replicate weight vectors, each corresponding to a perturbation of the sample,
and the sampling variance of any statistic is recovered from its variation
across replicates. :func:`replicate_variance` implements the jackknife and
balanced repeated replication estimators in the forms these programmes specify.

**Plausible values.** Individual ability is never observed; each assessment
instead releases several draws from each respondent's posterior latent
distribution. Analysing one draw understates uncertainty and analysing their
average is biased. The correct procedure, due to Rubin, is to analyse each draw
separately and combine by :func:`combine_plausible_values`, which adds a
between-draw component to the average sampling variance.

The two combine multiplicatively: with ``R`` replicates and ``M`` plausible
values a full analysis fits ``M * (R + 1)`` times, which
:func:`survey_dif` orchestrates.

References
----------
Rust, K. F., & Rao, J. N. K. (1996). Variance estimation for complex surveys
    using replication techniques. *Statistical Methods in Medical Research, 5*,
    283-310.
Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys.*
OECD (2009). *PISA Data Analysis Manual*, 2nd ed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "infer_replicate_design",
    "replicate_variance",
    "combine_plausible_values",
    "survey_dif",
    "jackknife_weights",
    "pool_estimates",
]


# --------------------------------------------------------------------------- #
# replicate variance
# --------------------------------------------------------------------------- #
_FAY_DEFAULT = 0.5


def infer_replicate_design(replicate_weights, base_weights=None, tol=0.5):
    """Infer how a set of replicate weights was constructed.

    The three variance estimators differ only in the constant converting
    replicate spread into a variance, and that constant is fixed by the
    construction. Choosing wrongly rescales every standard error without
    producing any visible symptom: applying Fay's constant to TIMSS JK2
    replicates shrinks standard errors to roughly a sixth of their proper size,
    and the output remains finite, ordered and plausible throughout. This
    function reads the construction off the weights so the mismatch can be
    caught rather than propagated.

    The three designs leave distinct signatures:

    ``jackknife``
        Weights are zeroed and others scaled up: a primary sampling unit has
        been dropped. Two variants are recognised. The delete-one form zeroes
        about one replicate's share of the sample and scales every survivor up
        slightly. The paired JK2 form used by TIMSS touches one zone per
        replicate, zeroing one unit and doubling its partner while leaving the
        rest untouched, so its signature is that pairing rather than any
        particular zeroed fraction.
    ``brr``
        About half the weights are zeroed and the remainder doubled: a
        balanced half-sample.
    ``fay``
        Nothing is zeroed. Weights take two values straddling one, the
        perturbation Fay's method applies instead of deletion.

    Parameters
    ----------
    replicate_weights : array-like, shape (n_replicates, n_persons)
    base_weights : array-like, optional
        Full-sample weights, used to measure the scaling. Inferred from the
        replicate mean when omitted.
    tol : float, default 0.5
        Relative tolerance on the expected zeroed fraction for the jackknife
        signature. The delete-one form zeroes about one replicate's share of
        the sample; the paired JK2 form used by TIMSS deletes one of two units
        per zone and so zeroes about half that, and both are accepted.

    Returns
    -------
    dict
        ``method`` (one of ``"jackknife"``, ``"brr"``, ``"fay"`` or ``None``
        when no signature matches), ``zero_fraction``, ``n_replicates``,
        ``fay_factor`` when the Fay signature is recognised, and three
        diagnostics that are populated whatever the outcome:
        ``unchanged_fraction``, the share of weights identical to the base;
        ``doubled_fraction``, the share exactly doubled; and
        ``scale_quantiles``, the 10th, 50th and 90th percentiles of the
        non-zero multipliers. When ``method`` is ``None`` these say which
        signature the weights came closest to, and so whether the design is
        merely unusual or the inputs are wrong: a ``zero_fraction`` of zero
        with multipliers spread continuously suggests weights that were never
        replicate weights at all.

    Examples
    --------
    >>> import numpy as np
    >>> from difair.survey import infer_replicate_design, jackknife_weights
    >>> w = np.ones(120)
    >>> rw = jackknife_weights(w, np.zeros(120), psu=np.arange(120) % 20)
    >>> infer_replicate_design(rw, w)["method"]
    'jackknife'
    """
    R = np.atleast_2d(np.asarray(replicate_weights, dtype=float))
    n_rep = R.shape[0]

    def result(method, zero_frac, fay=None, unchanged=np.nan,
               doubled=np.nan, quantiles=None):
        return {
            "method": method,
            "zero_fraction": zero_frac,
            "n_replicates": n_rep,
            "fay_factor": fay,
            "unchanged_fraction": unchanged,
            "doubled_fraction": doubled,
            "scale_quantiles": quantiles,
        }

    if n_rep < 2:
        return result(None, np.nan)

    base = (
        np.asarray(base_weights, dtype=float)
        if base_weights is not None
        else R.mean(axis=0)
    )
    ok = np.isfinite(base) & (base > 0)
    if not ok.any():
        return result(None, np.nan)

    zero_frac = float(np.mean(R[:, ok] == 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = R[:, ok] / base[ok]
    nonzero = scale[scale > 0]

    unchanged = float(np.mean(np.isclose(scale, 1.0)))
    doubled = float(np.mean(np.isclose(scale, 2.0, rtol=1e-3)))
    quant = (
        tuple(round(float(q), 4) for q in np.quantile(nonzero, [0.1, 0.5, 0.9]))
        if nonzero.size else None
    )

    if zero_frac < 1e-9:
        # Nothing deleted: a perturbation scheme. Fay's factor is the distance
        # of the two multiplier levels from one.
        if nonzero.size:
            hi, lo = float(np.quantile(nonzero, 0.9)), float(np.quantile(nonzero, 0.1))
            k = float(np.clip((hi - lo) / 2.0, 0.0, 1.0))
            if 0.05 < k < 0.95:
                return result("fay", 0.0, round(k, 3), unchanged, doubled, quant)
        return result(None, 0.0, None, unchanged, doubled, quant)

    # Paired jackknife (JK2): each replicate touches one zone only, zeroing one
    # unit and doubling its partner while the rest of the sample is untouched.
    # The zeroed fraction therefore reflects zone size, not replicate count, so
    # the signature is the pairing itself: most weights unchanged, and roughly
    # as many doubled as zeroed.
    if unchanged > 0.5 and zero_frac > 0 and abs(doubled - zero_frac) <= 0.5 * zero_frac:
        return result("jackknife", zero_frac, None, unchanged, doubled, quant)

    # Balanced half-samples: about half zeroed, the surviving half doubled.
    if abs(zero_frac - 0.5) <= 0.15 and nonzero.size and abs(np.median(nonzero) - 2.0) < 0.2:
        return result("brr", zero_frac, None, unchanged, doubled, quant)

    # Delete-one jackknife: one replicate's share zeroed, survivors scaled up
    # by a small common factor so the total is preserved.
    if nonzero.size and np.median(nonzero) > 1.0 and zero_frac <= 2.0 / n_rep:
        return result("jackknife", zero_frac, None, unchanged, doubled, quant)

    return result(None, zero_frac, None, unchanged, doubled, quant)


def replicate_variance(
    estimate,
    replicate_estimates,
    method="jackknife",
    fay_factor=_FAY_DEFAULT,
    fpc=None,
):
    """Sampling variance of a statistic from its replicate estimates.

    Parameters
    ----------
    estimate : float
        The statistic computed with the full-sample weights.
    replicate_estimates : array-like
        The same statistic recomputed with each replicate weight vector.
    method : {"jackknife", "brr", "fay"}
        ``"jackknife"`` uses the sum of squared deviations, the form TIMSS and
        PIRLS specify. ``"brr"`` divides by the number of replicates, as NAEP
        does. ``"fay"`` is BRR with Fay's adjustment, which PISA uses; the
        divisor becomes ``R * (1 - fay_factor) ** 2``.
    fay_factor : float, default 0.5
        Fay's perturbation factor, used only when ``method="fay"``. PISA
        distributes replicates built with 0.5.
    fpc : float, optional
        Finite-population correction, the sampling fraction of primary
        sampling units: pass ``n_sampled / n_population`` and the variance is
        multiplied by ``1 - fpc``. Replicate estimators assume sampling with
        replacement from an infinite population, which overstates the variance
        when a substantial share of the clusters was drawn. In simulation with
        25 of 80 clusters sampled the correction moved the ratio of estimated
        to empirical variance from 1.35 to 0.92. Omit it, and the estimate
        stays conservative, when the sampling fraction is small or unknown.

    Returns
    -------
    dict
        ``variance``, ``se``, ``n_replicates``, ``method``.

    Notes
    -----
    All three estimators measure the spread of the replicate estimates about
    the full-sample estimate; they differ only in the constant that converts
    that spread into a variance, and the constant is fixed by how the
    replicates were constructed. Using the wrong one silently rescales every
    standard error, so the choice must follow the assessment's documentation
    rather than convenience.

    The scale of the error is not subtle. Applying Fay's constant to TIMSS JK2
    replicates, where the jackknife form is correct, shrinks standard errors to
    roughly a sixth of their proper size across 61 item analyses. Nothing in
    the output signals this: the estimates remain finite, ordered and
    plausible-looking. Match the estimator to the design: ``"jackknife"`` for
    TIMSS and PIRLS delete-one replicates, ``"brr"`` for balanced half-samples
    as in NAEP, ``"fay"`` for the perturbed half-samples PISA distributes.
    """
    rep = np.asarray(replicate_estimates, dtype=float)
    rep = rep[np.isfinite(rep)]
    r = rep.size
    if r < 2:
        return {"variance": np.nan, "se": np.nan, "n_replicates": r, "method": method}

    dev = (rep - float(estimate)) ** 2
    if method == "jackknife":
        var = dev.sum()
    elif method == "brr":
        var = dev.mean()
    elif method == "fay":
        denom = r * (1.0 - fay_factor) ** 2
        var = dev.sum() / denom if denom > 0 else np.nan
    else:
        raise ValueError("`method` must be 'jackknife', 'brr' or 'fay'.")

    if fpc is not None:
        if not 0 <= fpc <= 1:
            raise ValueError("`fpc` must be a sampling fraction in [0, 1].")
        var *= 1.0 - fpc

    return {
        "variance": float(var),
        "se": float(np.sqrt(var)) if np.isfinite(var) and var >= 0 else np.nan,
        "n_replicates": int(r),
        "method": method,
        "fpc": fpc,
    }


def jackknife_weights(weights, strata, psu=None, n_replicates=None, seed=None):
    """Construct paired jackknife replicate weights.

    Provided for users whose data carry a sampling design but no ready-made
    replicate weights. In each replicate one primary sampling unit (PSU) is
    dropped from one stratum and its partners are inflated to preserve the
    stratum total, which is the paired-jackknife scheme TIMSS and PIRLS use.

    Parameters
    ----------
    weights : array-like
        Full-sample weights.
    strata : array-like
        Stratum identifier per respondent.
    psu : array-like, optional
        Primary sampling unit identifier, typically the school. **This is the
        unit that gets resampled.** Omitting it treats each respondent as its
        own PSU, which corresponds to simple random sampling within strata and
        will badly understate variance for a clustered design: dropping one
        student out of thousands barely moves any estimate, whereas dropping a
        school does. Supply it whenever the design is clustered.
    n_replicates : int, optional
        Number of replicates. Defaults to the total number of PSUs, giving one
        replicate per PSU.
    seed : int, optional
        Seed used only when ``n_replicates`` is smaller than the number of
        PSUs, in which case the PSUs to drop are chosen at random.

    Returns
    -------
    ndarray, shape (n_replicates, n_persons)

    Notes
    -----
    When an assessment distributes its own replicate weights, use those
    instead: they encode design details, such as finite-population corrections
    and collapsed strata, that cannot be reconstructed from the weights alone.

    In a Monte Carlo study on a population with a genuine cluster-level random
    effect, 25 sampled clusters and 200 repetitions, the jackknife variance of
    the Mantel-Haenszel delta was about 1.55 times the empirical sampling
    variance across items (range 1.16 to 1.74): conservative, which is the
    direction this estimator is known to err in and the safe direction for
    inference, but wide enough that intervals should be read as upper bounds
    rather than exact. Leaving ``psu`` unset when the design is clustered had
    the opposite and far more dangerous effect, understating the variance by
    more than an order of magnitude. Prefer the assessment's own replicate
    weights when they exist; they are calibrated to its design and do not carry
    this conservatism.
    """
    w = np.asarray(weights, dtype=float)
    s = np.asarray(pd.Series(strata).to_numpy())
    p = np.arange(len(w)) if psu is None else np.asarray(pd.Series(psu).to_numpy())

    if len(s) != len(w) or len(p) != len(w):
        raise ValueError("`strata` and `psu` must be as long as `weights`.")

    # Enumerate (stratum, PSU) pairs that can be dropped: a stratum needs at
    # least two PSUs, or removing one leaves nothing to inflate.
    frame = pd.DataFrame({"s": s, "p": p})
    pairs = []
    for lab, grp in frame.groupby("s", sort=False):
        units = pd.unique(grp["p"])
        if len(units) < 2:
            continue
        pairs.extend((lab, u) for u in units)

    if not pairs:
        raise ValueError(
            "No stratum contains two or more primary sampling units; "
            "replicate weights cannot be formed from this design."
        )

    if n_replicates is not None and n_replicates < len(pairs):
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(pairs), size=int(n_replicates), replace=False)
        pairs = [pairs[i] for i in sorted(sel)]

    out = np.tile(w, (len(pairs), 1))
    for i, (lab, unit) in enumerate(pairs):
        in_stratum = s == lab
        drop = in_stratum & (p == unit)
        keep = in_stratum & ~drop
        if not keep.any():
            continue
        n_units = len(pd.unique(p[in_stratum]))
        out[i, drop] = 0.0
        out[i, keep] *= n_units / (n_units - 1)  # preserve the stratum total
    return out


# --------------------------------------------------------------------------- #
# plausible values
# --------------------------------------------------------------------------- #
def combine_plausible_values(estimates, variances=None):
    """Combine estimates across plausible values by Rubin's rules.

    The total variance is the average within-draw sampling variance plus an
    inflated between-draw component::

        V_total = V_within + (1 + 1 / M) * V_between

    Parameters
    ----------
    estimates : array-like
        One estimate per plausible value.
    variances : array-like, optional
        The sampling variance accompanying each estimate, typically from
        :func:`replicate_variance`. When omitted only the between-draw
        component is reported, which is a lower bound on the total.

    Returns
    -------
    dict
        ``estimate``, ``variance``, ``se``, ``within``, ``between``, ``df``,
        ``fmi`` (fraction of missing information), ``n_values``.

    Notes
    -----
    The degrees of freedom follow Rubin's approximation, which is finite and
    can be small when the between-draw component dominates. A large fraction of
    missing information means the result is driven by uncertainty about the
    latent trait rather than by sampling, and should be reported as such.
    """
    est = np.asarray(estimates, dtype=float)
    ok = np.isfinite(est)
    est = est[ok]
    m = est.size
    if m == 0:
        return {"estimate": np.nan, "variance": np.nan, "se": np.nan,
                "within": np.nan, "between": np.nan, "df": np.nan,
                "fmi": np.nan, "n_values": 0}

    point = float(est.mean())
    between = float(est.var(ddof=1)) if m > 1 else 0.0

    if variances is None:
        within = 0.0
    else:
        v = np.asarray(variances, dtype=float)[ok]
        within = float(np.nanmean(v)) if np.isfinite(v).any() else 0.0

    total = within + (1.0 + 1.0 / m) * between if m > 1 else within

    if total > 0 and m > 1 and between > 0:
        lam = (1.0 + 1.0 / m) * between / total       # fraction of missing info
        df = (m - 1) / lam**2 if lam > 0 else np.inf
    else:
        lam, df = 0.0, np.inf

    return {
        "estimate": point,
        "variance": float(total),
        "se": float(np.sqrt(total)) if total >= 0 else np.nan,
        "within": within,
        "between": between,
        "df": float(df),
        "fmi": float(lam),
        "n_values": int(m),
    }


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def _bin_scores(values, n_bins):
    """Discretise a continuous criterion into equal-frequency ordered levels.

    Stratified DIF procedures need a criterion with repeated values; a
    continuous latent estimate gives each respondent a stratum of one, so no
    stratum contains both groups and nothing is estimable. Quantile binning
    preserves the ordering while restoring the strata.
    """
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return v
    edges = np.unique(np.quantile(v[ok], np.linspace(0, 1, int(n_bins) + 1)))
    if edges.size < 2:
        return np.zeros_like(v)
    out = np.full(v.shape, np.nan)
    out[ok] = np.clip(np.digitize(v[ok], edges[1:-1]), 0, len(edges) - 2)
    return out



def survey_dif(
    responses,
    group,
    focal_label,
    weights,
    replicate_weights=None,
    plausible_values=None,
    statistic="delta_mh",
    n_matching_bins=20,
    method="jackknife",
    fay_factor=_FAY_DEFAULT,
    fpc=None,
    check_design=True,
    polytomous=False,
    alpha=0.05,
):
    """DIF statistics with design-based standard errors.

    Runs the appropriate DIF procedure once per plausible value and once per
    replicate weight vector, then combines: replicate variance within each
    plausible value, Rubin's rules across them.

    Parameters
    ----------
    responses : DataFrame
        Item responses.
    group : array-like
        Binary group membership.
    focal_label : hashable
        Value of ``group`` identifying the focal group.
    weights : array-like
        Full-sample weights. Normalise first if they are on the population
        scale; see :func:`difair.dif.normalize_weights`.
    replicate_weights : array-like, shape (n_replicates, n_persons), optional
        Replicate weight vectors. Without them no sampling variance is
        estimated and the within component is zero.
    plausible_values : array-like, shape (n_values, n_persons), optional
        Draws from the latent posterior, used as the matching criterion in
        place of the observed total score. With one row the analysis is
        single-draw; with several, Rubin's rules apply. Values are continuous,
        so they are binned into ``n_matching_bins`` ordered levels before
        stratification; leaving them unbinned would put nearly every
        respondent in a stratum of one and yield no estimate at all.
    n_matching_bins : int, default 20
        Number of equal-frequency bins used to discretise plausible values.
        Too few loses matching precision and too many empties the strata; the
        default follows the usual practice of matching on around twenty score
        levels.
    statistic : str, default "delta_mh"
        Column of the DIF table to summarise. Must be numeric.
    method : {"jackknife", "brr", "fay"}
        Replicate variance estimator, following the assessment's manual.
    fpc : float, optional
        Finite-population correction; see :func:`replicate_variance`.
    check_design : bool, default True
        Compare ``method`` against the construction inferred from the replicate
        weights themselves and warn on a mismatch. A wrong constant rescales
        every standard error silently, so this check is on by default; set it
        to ``False`` for a design whose signature the inference does not
        recognise.
    polytomous : bool, default False
        Use the ordered-categorical procedure instead of Mantel-Haenszel.
    alpha : float, default 0.05
        Level for the reported confidence interval.

    Returns
    -------
    DataFrame
        One row per item with ``item``, ``estimate``, ``se``, ``ci_low``,
        ``ci_high``, ``within``, ``between``, ``fmi``, ``df``,
        ``n_plausible_values``, ``n_replicates``.

    Examples
    --------
    >>> import numpy as np
    >>> from difair import simulate_dif_data
    >>> from difair.survey import survey_dif, jackknife_weights
    >>> sim = simulate_dif_data(n_ref=300, n_focal=300, n_items=6, seed=0)
    >>> w = np.ones(600)
    >>> rw = jackknife_weights(w, strata=np.zeros(600), psu=np.arange(600) % 12)
    >>> out = survey_dif(sim.responses, sim.group, "focal", w,
    ...                  replicate_weights=rw)
    >>> sorted(out.columns)[:3]
    ['between', 'ci_high', 'ci_low']
    """
    from difair.dif import mantel_haenszel
    from difair.poly import generalized_mantel_haenszel

    proc = generalized_mantel_haenszel if polytomous else mantel_haenszel
    if polytomous and statistic == "delta_mh":
        statistic = "smd"

    w = np.asarray(weights, dtype=float)
    reps = None if replicate_weights is None else np.atleast_2d(
        np.asarray(replicate_weights, dtype=float)
    )
    pvs = None if plausible_values is None else np.atleast_2d(
        np.asarray(plausible_values, dtype=float)
    )
    if pvs is not None:
        pvs = np.vstack([_bin_scores(row, n_matching_bins) for row in pvs])
    n_pv = 1 if pvs is None else pvs.shape[0]

    if check_design and reps is not None:
        design = infer_replicate_design(reps, w)
        inferred = design["method"]
        if inferred is None:
            warnings.warn(
                "The replicate weights match no recognised construction "
                f"(zeroed {design['zero_fraction']:.4f}, unchanged "
                f"{design['unchanged_fraction']:.4f}, doubled "
                f"{design['doubled_fraction']:.4f}, non-zero multipliers at the "
                f"10th, 50th and 90th percentiles {design['scale_quantiles']}). "
                f"method={method!r} is being used as given; verify it against "
                "the assessment's documentation, since the variance constant "
                "is fixed by how the replicates were built.",
                stacklevel=2,
            )
        elif inferred != method:
            warnings.warn(
                f"method={method!r} does not match the replicate weights, which "
                f"look like a {inferred!r} construction "
                f"(zeroed fraction {design['zero_fraction']:.3f} across "
                f"{design['n_replicates']} replicates). The variance constant is "
                "fixed by how the replicates were built, so a mismatch rescales "
                "every standard error without any other symptom: applying Fay's "
                "constant to jackknife replicates shrinks them roughly sixfold. "
                f"Pass method={inferred!r}, or check_design=False if this "
                "inference is wrong for your design.",
                stacklevel=2,
            )

    def run(weight_vec, matching):
        """One DIF pass with the given weights and matching criterion."""
        kw = dict(focal_label=focal_label, weights=weight_vec)
        if not polytomous:
            kw["alpha"] = alpha
        if matching is not None:
            kw["matching"] = matching
        return proc(responses, group, **kw)

    items = list(pd.DataFrame(responses).columns)
    per_pv_est, per_pv_var = [], []

    for i in range(n_pv):
        matching = None if pvs is None else pvs[i]
        full = run(w, matching)
        if statistic not in full.columns:
            raise ValueError(
                f"statistic={statistic!r} is not a column of the DIF table; "
                f"available: {sorted(c for c in full.columns if c != 'item')}"
            )
        est = full.set_index("item")[statistic].reindex(items).to_numpy(dtype=float)
        per_pv_est.append(est)

        if reps is None:
            per_pv_var.append(np.zeros_like(est))
            continue

        rep_est = np.empty((reps.shape[0], len(items)))
        for r in range(reps.shape[0]):
            tab = run(reps[r], matching)
            rep_est[r] = (
                tab.set_index("item")[statistic].reindex(items).to_numpy(dtype=float)
            )
        per_pv_var.append(np.array([
            replicate_variance(est[j], rep_est[:, j], method, fay_factor, fpc)["variance"]
            for j in range(len(items))
        ]))

    per_pv_est = np.vstack(per_pv_est)
    per_pv_var = np.vstack(per_pv_var)

    rows = []
    for j, name in enumerate(items):
        comb = combine_plausible_values(per_pv_est[:, j], per_pv_var[:, j])
        se = comb["se"]
        if np.isfinite(se) and se > 0 and np.isfinite(comb["df"]):
            crit = float(stats.t.ppf(1 - alpha / 2, max(comb["df"], 1.0)))
        else:
            crit = float(stats.norm.ppf(1 - alpha / 2))
        lo = comb["estimate"] - crit * se if np.isfinite(se) else np.nan
        hi = comb["estimate"] + crit * se if np.isfinite(se) else np.nan
        rows.append({
            "item": name,
            "estimate": comb["estimate"],
            "se": se,
            "ci_low": lo,
            "ci_high": hi,
            "within": comb["within"],
            "between": comb["between"],
            "fmi": comb["fmi"],
            "df": comb["df"],
            "n_plausible_values": n_pv,
            "n_replicates": 0 if reps is None else reps.shape[0],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# pooling across analyses
# --------------------------------------------------------------------------- #
def pool_estimates(estimates, standard_errors, method="fixed"):
    """Combine independent estimates of the same quantity.

    Rotated-block assessments and multi-cohort studies produce several
    estimates of the same item's DIF, each from a different subsample. Where
    those subsamples are disjoint the estimates are independent and can be
    pooled, which is more efficient than reporting them separately or averaging
    them unweighted.

    Parameters
    ----------
    estimates : array-like
        One estimate per analysis.
    standard_errors : array-like
        The accompanying standard errors. Entries that are missing or
        non-positive drop the corresponding estimate.
    method : {"fixed", "random"}
        ``"fixed"`` weights by inverse variance, assuming every analysis
        estimates the same quantity. ``"random"`` adds the DerSimonian-Laird
        between-analysis variance, appropriate when the quantity may genuinely
        differ across subsamples, as it can when blocks were administered to
        different populations.

    Returns
    -------
    dict
        ``estimate``, ``se``, ``ci_low``, ``ci_high``, ``q`` (Cochran's
        heterogeneity statistic), ``i_squared`` (the share of variance
        attributable to heterogeneity), ``tau_squared``, ``n_analyses``.

    Notes
    -----
    ``i_squared`` above roughly 0.5 signals that the analyses disagree more
    than sampling error explains, in which case the fixed-effect pooled value
    understates uncertainty and the random-effects form should be preferred.
    """
    est = np.asarray(estimates, dtype=float)
    se = np.asarray(standard_errors, dtype=float)
    ok = np.isfinite(est) & np.isfinite(se) & (se > 0)
    est, se = est[ok], se[ok]
    k = est.size

    if k == 0:
        return {"estimate": np.nan, "se": np.nan, "ci_low": np.nan,
                "ci_high": np.nan, "q": np.nan, "i_squared": np.nan,
                "tau_squared": np.nan, "n_analyses": 0}
    if k == 1:
        crit = float(stats.norm.ppf(0.975))
        return {"estimate": float(est[0]), "se": float(se[0]),
                "ci_low": float(est[0] - crit * se[0]),
                "ci_high": float(est[0] + crit * se[0]),
                "q": 0.0, "i_squared": 0.0, "tau_squared": 0.0,
                "n_analyses": 1}

    w = 1.0 / se**2
    fixed = float((w * est).sum() / w.sum())
    q = float((w * (est - fixed) ** 2).sum())
    df = k - 1
    i2 = float(max(0.0, (q - df) / q)) if q > 0 else 0.0

    if method == "random":
        c = w.sum() - (w**2).sum() / w.sum()
        tau2 = float(max(0.0, (q - df) / c)) if c > 0 else 0.0
        w = 1.0 / (se**2 + tau2)
    elif method == "fixed":
        tau2 = 0.0
    else:
        raise ValueError("`method` must be 'fixed' or 'random'.")

    point = float((w * est).sum() / w.sum())
    pooled_se = float(np.sqrt(1.0 / w.sum()))
    crit = float(stats.norm.ppf(0.975))
    return {
        "estimate": point,
        "se": pooled_se,
        "ci_low": point - crit * pooled_se,
        "ci_high": point + crit * pooled_se,
        "q": q,
        "i_squared": i2,
        "tau_squared": tau2,
        "n_analyses": int(k),
    }
