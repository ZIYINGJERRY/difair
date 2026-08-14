"""Cross-country pooling and replicate-design inference on TIMSS 2019.

The v0.6 review left two capabilities validated only against synthetic or
analytic checks: the inference that reads a replicate construction off the
weights, and the meta-analytic pooling of independent estimates. Both are
exercised here on released data.

**Replicate-design inference.** Each participating country draws its own
sample, so the number of jackknife zones, the students per zone and the
stratification all differ between country files. Running the inference across
several countries therefore tests it against genuinely distinct designs rather
than against variations of one simulator. This matters: the first version of
the inference recognised every synthetic design and none of the real ones,
because it had been written against the package's own generator.

**Pooling.** TIMSS 2019 assigns each item to a single block, so no item yields
two estimates within a country. Across countries it does: the same item is
administered in every participating country, giving one design-based estimate
per country of the same quantity, namely the sex DIF of that item. Whether
those estimates agree is itself the question of interest — a large I-squared
means the item functions differently between educational systems, which is
what cross-national DIF research looks for.

Usage
-----
    python examples/timss_pooling.py --sav path/to/*.sav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from difair.dif import mantel_haenszel, normalize_weights  # noqa: E402
from difair.survey import (  # noqa: E402
    infer_replicate_design,
    pool_estimates,
    survey_dif,
)
from timss_validation import (  # noqa: E402
    ITEM_RE,
    _score_item,
    jk2_replicates,
)

warnings.filterwarnings("ignore")

OUT = Path(__file__).resolve().parent.parent / "validation_results"


def load_country(path, min_coverage=0.9, min_items=5):
    """Score one country file into a per-block response structure."""
    import pyreadstat

    df, meta = pyreadstat.read_sav(str(path))
    labels = meta.column_names_to_labels
    items = [
        c for c in df.columns
        if ITEM_RE.match(c) and df[c].notna().any()
        and np.allclose(df[c].dropna().values, np.round(df[c].dropna().values))
    ]

    blocks = {}
    for c in items:
        key = tuple(sorted(df.loc[df[c].notna(), "IDBOOK"].dropna().unique()))
        if key:
            blocks.setdefault(key, []).append(c)

    out = []
    for booklets, cols in blocks.items():
        sub = df[df["IDBOOK"].isin(booklets)]
        cov = sub[cols].notna().mean()
        keep = cov[cov >= min_coverage].index.tolist()

        scored = {}
        for c in keep:
            s = _score_item(sub[c], labels.get(c, ""))
            if s is not None and s.notna().mean() >= min_coverage and 0 < s.mean() < 1:
                scored[c] = s
        resp = pd.DataFrame(scored).dropna()
        if resp.shape[1] < min_items:
            continue
        rows = sub.loc[resp.index]
        out.append({
            "responses": resp.astype(int),
            "group": rows["ITSEX"].to_numpy(),
            "weights": rows["TOTWGT"].to_numpy(float),
            "jkzone": rows["JKZONE"].to_numpy(),
            "jkrep": rows["JKREP"].to_numpy(),
            "n_students": len(rows),
        })
    return out, df


def check_designs(paths):
    """Does the inference recognise each country's own sampling design?"""
    import pyreadstat

    rows = []
    for path in paths:
        df, _ = pyreadstat.read_sav(str(path))
        if not {"TOTWGT", "JKZONE", "JKREP"} <= set(df.columns):
            continue
        w = normalize_weights(df["TOTWGT"].to_numpy(float))
        reps, zones = jk2_replicates(w, df["JKZONE"].to_numpy(), df["JKREP"].to_numpy())
        got = infer_replicate_design(reps, w)
        rows.append({
            "file": Path(path).stem,
            "n_students": len(df),
            "n_zones": len(zones),
            "students_per_zone": round(len(df) / max(len(zones), 1), 1),
            "zero_fraction": round(got["zero_fraction"], 5),
            "inferred": got["method"],
            "correct": got["method"] == "jackknife",
        })
    return pd.DataFrame(rows)


def country_estimates(paths, statistic="delta_mh"):
    """Design-based per-item DIF estimates, one row per country and item."""
    rows = []
    for path in paths:
        name = Path(path).stem
        blocks, _ = load_country(path)
        for blk in blocks:
            w = normalize_weights(blk["weights"])
            reps, _ = jk2_replicates(w, blk["jkzone"], blk["jkrep"])
            res = survey_dif(blk["responses"], blk["group"], 1.0, w,
                             replicate_weights=reps, method="jackknife",
                             statistic=statistic)
            wtd = mantel_haenszel(blk["responses"], blk["group"],
                                  focal_label=1.0, weights=w)
            for _, r in res.iterrows():
                rows.append({
                    "country": name,
                    "item": r["item"],
                    "estimate": r["estimate"],
                    "se": r["se"],
                    "ets_class": wtd.loc[wtd.item == r["item"], "ets_class"].iloc[0],
                    "n_students": blk["n_students"],
                })
        print(f"  {name}: {sum(b['responses'].shape[1] for b in blocks)} item "
              f"estimates from {len(blocks)} blocks")
    return pd.DataFrame(rows)


def main(sav, statistic="delta_mh"):
    OUT.mkdir(exist_ok=True)
    paths = [Path(p) for p in sav]

    print(f"Replicate-design inference across {len(paths)} country files")
    designs = check_designs(paths)
    print(designs.to_string(index=False))
    designs.to_csv(OUT / "timss_design_inference.csv", index=False)
    print(f"  correctly identified: {int(designs.correct.sum())} of {len(designs)}\n")

    print("Design-based per-item estimates")
    est = country_estimates(paths, statistic)
    if est.empty:
        raise SystemExit("No usable estimates were produced.")

    shared = est.groupby("item").country.nunique()
    shared = shared[shared >= 2].index
    est = est[est.item.isin(shared)]
    print(f"\n{len(shared)} items estimated in two or more countries")

    pooled = []
    for item, g in est.groupby("item"):
        ok = g[np.isfinite(g.se) & (g.se > 0)]
        if len(ok) < 2:
            continue
        fixed = pool_estimates(ok.estimate, ok.se, method="fixed")
        rand = pool_estimates(ok.estimate, ok.se, method="random")
        pooled.append({
            "item": item, "n_countries": len(ok),
            "estimate_fixed": fixed["estimate"], "se_fixed": fixed["se"],
            "estimate_random": rand["estimate"], "se_random": rand["se"],
            "q": fixed["q"], "i_squared": fixed["i_squared"],
            "tau_squared": rand["tau_squared"],
            "ci_low": rand["ci_low"], "ci_high": rand["ci_high"],
        })
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(OUT / "timss_pooled_cross_country.csv", index=False)

    if pooled.empty:
        print("No item had two usable estimates.")
        return pooled

    het = pooled.i_squared
    print(f"  pooled {len(pooled)} items across countries")
    print(f"  heterogeneity I-squared: median {het.median():.3f}, "
          f"{int((het > 0.5).sum())} of {len(pooled)} above 0.5")
    print(f"  random-effects SE exceeds fixed-effect SE for "
          f"{int((pooled.se_random > pooled.se_fixed + 1e-12).sum())} of {len(pooled)} items")
    consistent = pooled[(pooled.ci_low > 0) | (pooled.ci_high < 0)]
    print(f"  {len(consistent)} items show sex DIF consistent across countries "
          f"(pooled interval excludes zero)")

    print(f"\nresults -> {OUT}")
    return pooled


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sav", required=True, nargs="+")
    ap.add_argument("--statistic", default="delta_mh")
    main(**vars(ap.parse_args()))
