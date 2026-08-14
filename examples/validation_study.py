"""Validation study for difair.

Runs the four experiments reported in the accompanying paper:

1. Recovery -- detection power and Type I error across DIF magnitude and
   sample size.
2. Impact robustness -- Type I error when the groups genuinely differ in
   ability but no item carries DIF.
3. Scalability -- wall-clock time as test length and sample size grow.
4. Attribution recovery -- whether Shapley stage attribution tracks the
   pipeline stage where disparity was planted.

Usage
-----
    python examples/validation_study.py [--quick]

Results are written to ``validation_results/`` as CSV.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from difair.dif import logistic_dif, mantel_haenszel, standardization
from difair.pipeline import attribute_stages
from difair.poly import generalized_mantel_haenszel
from difair.survey import jackknife_weights, replicate_variance
from difair.simulate import (
    simulate_dif_data,
    simulate_pipeline_data,
    simulate_poly_dif_data,
)

warnings.filterwarnings("ignore")

OUT = Path(__file__).resolve().parent.parent / "validation_results"
N_ITEMS, N_DIF = 20, 4


# --------------------------------------------------------------------------- #
def experiment_recovery(n_rep, magnitudes, sample_sizes):
    """Power and Type I error for MH and standardization."""
    rows = []
    for n in sample_sizes:
        for mag in magnitudes:
            hit_mh = fa_mh = hit_std = fa_std = 0
            delta_sum = 0.0
            for r in range(n_rep):
                sim = simulate_dif_data(
                    n_ref=n, n_focal=n, n_items=N_ITEMS, n_dif_items=N_DIF,
                    dif_magnitude=mag, seed=10_000 + r,
                )
                truth = set(sim.dif_items)
                clean = set(sim.responses.columns) - truth

                mh = mantel_haenszel(sim.responses, sim.group, focal_label="focal")
                flagged = set(mh.loc[mh.ets_class.isin(["B", "C"]), "item"])
                hit_mh += len(flagged & truth)
                fa_mh += len(flagged & clean)
                delta_sum += abs(mh.loc[mh.item.isin(truth), "delta_mh"].mean())

                st = standardization(sim.responses, sim.group, focal_label="focal")
                fs = set(st.loc[st.std_class != "negligible", "item"])
                hit_std += len(fs & truth)
                fa_std += len(fs & clean)

            n_true = N_DIF * n_rep
            n_clean = (N_ITEMS - N_DIF) * n_rep
            rows.append({
                "n_per_group": n,
                "dif_magnitude": mag,
                "mh_power": hit_mh / n_true,
                "mh_type1": fa_mh / n_clean,
                "std_power": hit_std / n_true,
                "std_type1": fa_std / n_clean,
                "mean_abs_delta": delta_sum / n_rep,
                "n_replications": n_rep,
            })
            print(f"  [recovery] N={n:<5} delta={mag:.2f}  "
                  f"MH power={rows[-1]['mh_power']:.3f} type1={rows[-1]['mh_type1']:.3f}")
    return pd.DataFrame(rows)


def experiment_purification(n_rep, magnitudes, n):
    """Does purifying the matching score control the contamination effect?

    When DIF items contribute to the matching total, clean items are compared
    on a criterion that is itself biased, inflating the false-flag rate as DIF
    grows. Purification recomputes the total from unflagged items only.
    """
    from difair.dif import purify_matching_score

    rows = []
    for mag in magnitudes:
        raw_fa = pur_fa = raw_hit = pur_hit = 0
        for r in range(n_rep):
            sim = simulate_dif_data(
                n_ref=n, n_focal=n, n_items=N_ITEMS, n_dif_items=N_DIF,
                dif_magnitude=mag, seed=50_000 + r,
            )
            truth = set(sim.dif_items)
            clean = set(sim.responses.columns) - truth

            raw = mantel_haenszel(sim.responses, sim.group, focal_label="focal")
            f_raw = set(raw.loc[raw.ets_class.isin(["B", "C"]), "item"])
            raw_fa += len(f_raw & clean)
            raw_hit += len(f_raw & truth)

            match, _ = purify_matching_score(sim.responses, sim.group, focal_label="focal")
            pur = mantel_haenszel(sim.responses, sim.group, focal_label="focal",
                                  matching=match)
            f_pur = set(pur.loc[pur.ets_class.isin(["B", "C"]), "item"])
            pur_fa += len(f_pur & clean)
            pur_hit += len(f_pur & truth)

        n_clean = (N_ITEMS - N_DIF) * n_rep
        n_true = N_DIF * n_rep
        rows.append({
            "dif_magnitude": mag,
            "type1_raw": raw_fa / n_clean,
            "type1_purified": pur_fa / n_clean,
            "power_raw": raw_hit / n_true,
            "power_purified": pur_hit / n_true,
            "n_replications": n_rep,
        })
        print(f"  [purify]   delta={mag:.2f}  type1 {rows[-1]['type1_raw']:.3f} -> "
              f"{rows[-1]['type1_purified']:.3f}   power {rows[-1]['power_raw']:.3f} -> "
              f"{rows[-1]['power_purified']:.3f}")
    return pd.DataFrame(rows)


def experiment_logistic(n_rep, magnitudes, n):
    """Logistic-regression DIF: uniform vs non-uniform separation."""
    rows = []
    for kind, uni, non in (("uniform", 0.8, 0.0), ("nonuniform", 0.0, 0.9), ("none", 0.0, 0.0)):
        p_uni = p_non = 0
        for r in range(n_rep):
            sim = simulate_dif_data(
                n_ref=n, n_focal=n, n_items=N_ITEMS, n_dif_items=N_DIF,
                dif_magnitude=uni, nonuniform_magnitude=non, seed=20_000 + r,
            )
            res = logistic_dif(sim.responses, sim.group, focal_label="focal")
            hit = res[res.item.isin(sim.dif_items)]
            p_uni += (hit.p_uniform < 0.05).sum()
            p_non += (hit.p_nonuniform < 0.05).sum()
        tot = N_DIF * n_rep
        rows.append({
            "planted": kind,
            "rejection_rate_uniform_test": p_uni / tot,
            "rejection_rate_nonuniform_test": p_non / tot,
            "n_replications": n_rep,
        })
        print(f"  [logistic] planted={kind:<11} uniform={rows[-1]['rejection_rate_uniform_test']:.3f} "
              f"nonuniform={rows[-1]['rejection_rate_nonuniform_test']:.3f}")
    return pd.DataFrame(rows)


def experiment_impact(n_rep, impacts, n):
    """Genuine ability differences must not be read as DIF."""
    rows = []
    for imp in impacts:
        fa = 0
        for r in range(n_rep):
            sim = simulate_dif_data(
                n_ref=n, n_focal=n, n_items=N_ITEMS, n_dif_items=0,
                impact=imp, seed=30_000 + r,
            )
            mh = mantel_haenszel(sim.responses, sim.group, focal_label="focal")
            fa += mh.ets_class.isin(["B", "C"]).sum()
        rows.append({
            "impact": imp,
            "false_flag_rate": fa / (N_ITEMS * n_rep),
            "n_replications": n_rep,
        })
        print(f"  [impact]   impact={imp:.2f}  false-flag rate={rows[-1]['false_flag_rate']:.4f}")
    return pd.DataFrame(rows)


def experiment_polytomous(n_rep, magnitudes, n, n_categories=5):
    """Recovery for ordered categorical items via the generalized MH test."""
    rows = []
    for mag in magnitudes:
        hit = fa = 0
        smd_sum = 0.0
        for r in range(n_rep):
            sim = simulate_poly_dif_data(
                n_ref=n, n_focal=n, n_items=N_ITEMS, n_categories=n_categories,
                n_dif_items=N_DIF, dif_magnitude=mag, seed=60_000 + r,
            )
            truth = set(sim.dif_items)
            clean = set(sim.responses.columns) - truth
            res = generalized_mantel_haenszel(sim.responses, sim.group,
                                              focal_label="focal")
            flagged = set(res.loc[res.smd_class.isin(["B", "C"]), "item"])
            hit += len(flagged & truth)
            fa += len(flagged & clean)
            smd_sum += abs(res.loc[res.item.isin(truth), "smd"].mean())
        rows.append({
            "dif_magnitude": mag,
            "n_categories": n_categories,
            "power": hit / (N_DIF * n_rep),
            "type1": fa / ((N_ITEMS - N_DIF) * n_rep),
            "mean_abs_smd": smd_sum / n_rep,
            "n_replications": n_rep,
        })
        print(f"  [poly]     delta={mag:.2f}  power={rows[-1]['power']:.3f} "
              f"type1={rows[-1]['type1']:.3f}  |SMD|={rows[-1]['mean_abs_smd']:.3f}")
    return pd.DataFrame(rows)


def experiment_survey_variance(n_rep, n_clusters, n_sampled, seed0=70_000):
    """Does replicate-weight variance recover the true sampling variance?

    Builds a population with a genuine cluster-level random effect, draws
    ``n_rep`` cluster samples to obtain the empirical sampling variance of the
    Mantel-Haenszel delta, then estimates that variance from a single sample by
    jackknife replication. The ratio of the two is the quantity of interest.
    """
    rng = np.random.default_rng(seed0)
    per, J = 150, 8
    cl_ability = rng.normal(0, 0.5, n_clusters)
    cl_dif = rng.normal(0, 0.35, n_clusters)
    a = rng.uniform(0.8, 1.6, J)
    b = rng.uniform(-1, 1, J)

    resp, grp, cid = [], [], []
    for s_ in range(n_clusters):
        th = rng.normal(cl_ability[s_], 1.0, per)
        foc = rng.random(per) < 0.5
        bb = np.tile(b, (per, 1)).astype(float)
        bb[:, :2] += np.where(foc, 0.8 + cl_dif[s_], 0.0)[:, None]
        p = 1 / (1 + np.exp(-a * (th[:, None] - bb)))
        resp.append((rng.random(p.shape) < p).astype(int))
        grp.append(foc)
        cid.append(np.full(per, s_))
    U = pd.DataFrame(np.vstack(resp), columns=[f"i{j+1}" for j in range(J)])
    G, C = np.concatenate(grp), np.concatenate(cid)

    emp = []
    for _ in range(n_rep):
        pick = rng.choice(n_clusters, n_sampled, replace=False)
        idx = np.flatnonzero(np.isin(C, pick))
        emp.append(mantel_haenszel(U.iloc[idx], G[idx], focal_label=True).delta_mh.to_numpy())
    emp_var = np.vstack(emp).var(axis=0, ddof=1)

    pick = rng.choice(n_clusters, n_sampled, replace=False)
    idx = np.flatnonzero(np.isin(C, pick))
    w = np.ones(len(idx))
    rw = jackknife_weights(w, np.zeros(len(idx)), psu=C[idx])
    full = mantel_haenszel(U.iloc[idx], G[idx], focal_label=True, weights=w).delta_mh.to_numpy()
    rep = np.vstack([
        mantel_haenszel(U.iloc[idx], G[idx], focal_label=True, weights=rw[i]).delta_mh.to_numpy()
        for i in range(rw.shape[0])
    ])
    n = rw.shape[0]
    jk = np.array([
        replicate_variance(full[j], rep[:, j], "jackknife")["variance"] for j in range(J)
    ]) * (n - 1) / n

    ratio = jk / emp_var
    print(f"  [survey]   clusters={n_sampled}  median ratio={np.median(ratio):.2f}  "
          f"range {ratio.min():.2f}-{ratio.max():.2f}")
    return pd.DataFrame({
        "item": U.columns, "empirical_variance": emp_var,
        "jackknife_variance": jk, "ratio": ratio,
        "n_clusters_sampled": n_sampled, "n_repetitions": n_rep,
    })


def experiment_scalability(item_grid, person_grid, n_rep=3):
    """Wall-clock time as the problem grows."""
    rows = []
    for n_items in item_grid:
        for n in person_grid:
            sim = simulate_dif_data(
                n_ref=n, n_focal=n, n_items=n_items,
                n_dif_items=max(1, n_items // 5), seed=7,
            )
            times = []
            for _ in range(n_rep):
                t0 = time.perf_counter()
                mantel_haenszel(sim.responses, sim.group, focal_label="focal")
                times.append(time.perf_counter() - t0)
            rows.append({
                "n_items": n_items,
                "n_total_persons": 2 * n,
                "seconds_median": float(np.median(times)),
                "seconds_per_item": float(np.median(times) / n_items),
            })
            print(f"  [scale]    items={n_items:<4} persons={2*n:<7} "
                  f"{rows[-1]['seconds_median']:.3f}s")
    return pd.DataFrame(rows)


def experiment_attribution(n_rep, conditions):
    """Does the Shapley share follow the stage where disparity was planted?"""
    rows = []
    for name, kw in conditions.items():
        acc = {"item": [], "sampling": [], "model": []}
        base, resid = [], []
        for r in range(n_rep):
            d = simulate_pipeline_data(n_ref=1200, n_focal=1200, seed=40_000 + r, **kw)
            res = attribute_stages(
                d["responses"], d["group"], "focal", d["outcome"],
                dif_items=d["dif_items"], proxy=d["proxy"],
                train_mask=d["train_mask"], train_outcome=d["outcome_observed"],
                seed=r,
            )
            s = res.summary().set_index("stage")["shapley_value"]
            for k in acc:
                acc[k].append(float(s.get(k, 0.0)))
            base.append(res.baseline_gap)
            resid.append(res.residual_gap)
        row = {"condition": name, "n_replications": n_rep,
               "baseline_gap": float(np.mean(base)),
               "residual_gap": float(np.mean(resid))}
        total = sum(np.mean(v) for v in acc.values())
        # Mirror AttributionResult.summary: below the reporting threshold the
        # share is a quotient of two near-zero quantities and carries no
        # information, so report absolute Shapley values instead.
        reportable = abs(total) > 0.02
        for k, v in acc.items():
            row[f"shapley_{k}"] = float(np.mean(v))
            row[f"share_{k}"] = float(np.mean(v) / total) if reportable else np.nan
        rows.append(row)
        if reportable:
            print(f"  [attrib]   {name:<24} item={row['share_item']:+.3f} "
                  f"model={row['share_model']:+.3f} sampling={row['share_sampling']:+.3f}")
        else:
            print(f"  [attrib]   {name:<24} shares suppressed (explained gap "
                  f"{total:+.4f}); |shapley| max "
                  f"{max(abs(row[f'shapley_{k}']) for k in acc):.4f}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main(quick=False):
    OUT.mkdir(exist_ok=True)
    t_start = time.perf_counter()

    reps = 40 if quick else 200
    reps_lr = 15 if quick else 60
    reps_at = 10 if quick else 40

    print("1/8 recovery study")
    rec = experiment_recovery(reps, [0.0, 0.25, 0.5, 0.75, 1.0], [500, 1500])
    rec.to_csv(OUT / "recovery.csv", index=False)

    print("2/8 logistic separation")
    lr = experiment_logistic(reps_lr, None, 1500)
    lr.to_csv(OUT / "logistic.csv", index=False)

    print("3/8 purification")
    pur = experiment_purification(max(reps // 4, 20), [0.5, 0.75, 1.0], 500)
    pur.to_csv(OUT / "purification.csv", index=False)

    print("4/8 impact robustness")
    imp = experiment_impact(reps, [0.0, 0.5, 1.0], 1500)
    imp.to_csv(OUT / "impact.csv", index=False)

    print("5/8 polytomous recovery")
    poly = experiment_polytomous(max(reps // 4, 20), [0.0, 0.3, 0.6, 0.9], 1000)
    poly.to_csv(OUT / "polytomous.csv", index=False)

    print("6/8 survey variance recovery")
    surv = experiment_survey_variance(100 if quick else 200, 80, 25)
    surv.to_csv(OUT / "survey_variance.csv", index=False)

    print("7/8 scalability")
    sc = experiment_scalability([10, 20, 40, 80], [1000, 5000, 25000] if not quick else [1000, 5000])
    sc.to_csv(OUT / "scalability.csv", index=False)

    print("8/8 attribution recovery")
    conds = {
        "item-dominant": dict(dif_magnitude=1.2, label_bias=0.05, proxy_strength=0.2,
                              undersample_focal=0.95),
        "model-dominant": dict(dif_magnitude=0.1, label_bias=0.40, proxy_strength=0.9,
                               undersample_focal=0.95),
        "balanced": dict(dif_magnitude=0.8, label_bias=0.25, proxy_strength=0.7,
                         undersample_focal=0.60),
        "no-planted-disparity": dict(dif_magnitude=0.0, label_bias=0.0, proxy_strength=0.1,
                     undersample_focal=1.0),
    }
    at = experiment_attribution(reps_at, conds)
    at.to_csv(OUT / "attribution.csv", index=False)

    meta = {
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
        "quick_mode": quick,
        "n_items": N_ITEMS,
        "n_dif_items": N_DIF,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone in {meta['elapsed_seconds']}s -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer replications")
    main(**vars(ap.parse_args()))
