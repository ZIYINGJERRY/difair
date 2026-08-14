"""Validation of the design-based inference layer on TIMSS 2019 data.

The v0.3 review noted that :mod:`difair.survey` had been checked only against
simulated populations whose sampling variance was known by construction, and
that applying it to a released international assessment was the strongest
remaining test. This script does that.

TIMSS 2019 grade 8 student achievement files carry everything the module needs:
item-level responses, the final student weight ``TOTWGT``, the JK2 jackknife
design in ``JKZONE`` and ``JKREP``, five plausible values for mathematics, and
a sex variable to serve as the grouping factor.

Analysis is organised by *block* rather than by booklet. TIMSS rotates item
blocks across booklets, and in the 2019 grade 8 design each block appears in
exactly two of the fourteen booklets. Restricting to a single booklet therefore
discards half the students who saw a given item; grouping items by the set of
booklets containing them recovers them, roughly doubling the sample per block
without any loss of completeness. Blocks are analysed separately and their
per-item results concatenated, since no student sees every block.

Two properties are checked, neither of which the simulation study could reach.
First, whether replicate weights reconstructed from the published ``JKZONE``
and ``JKREP`` reproduce the design the assessment intends. Second, whether
standard errors computed under that design are materially larger than the ones
a naive simple-random-sampling analysis would report, which is the entire
reason the machinery exists.

Scoring follows the TIMSS convention. Multiple-choice items are keyed from the
correct option recorded in each variable's label; constructed-response items
score 1 for codes in the twenties, which denote full credit, and 0 otherwise.

Usage
-----
    python examples/timss_validation.py --sav path/to/bsageom7.sav
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from difair.dif import mantel_haenszel, normalize_weights
from difair.survey import infer_replicate_design, pool_estimates, survey_dif

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "validation_results"

KEY_RE = re.compile(r"\(([A-E1-9])\)\s*$")
ITEM_RE = re.compile(r"^ME\d")


def _score_item(values, label):
    """Score one item to 0/1 following the TIMSS conventions."""
    v = pd.Series(values).astype(float)
    observed = set(v.dropna().unique())

    # Multiple choice: options are coded 1-4 (occasionally 5), the key is the
    # letter in parentheses at the end of the variable label.
    if observed and observed <= {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0}:
        m = KEY_RE.search(label or "")
        if not m:
            return None
        token = m.group(1)
        key = float(ord(token) - ord("A") + 1) if token.isalpha() else float(token)
        return (v == key).astype(float).where(v.notna())

    # Constructed response: codes in the twenties are full credit.
    if observed and max(observed) >= 10:
        return ((v >= 20) & (v < 30)).astype(float).where(v.notna())
    return None


def _block_key(df, item):
    """The set of booklets in which an item appears, identifying its block."""
    return tuple(sorted(df.loc[df[item].notna(), "IDBOOK"].dropna().unique()))


def load_timss(paths, min_coverage=0.9):
    """Return scored responses, group, weights, design columns and PVs.

    Accepts one or more country files. Pooling countries enlarges the sample
    and is itself the setting international DIF studies care about, but the
    jackknife zones are numbered within country, so they are made unique before
    the replicate weights are built.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    try:
        import pyreadstat
    except ImportError:  # pragma: no cover - dependency is optional
        sys.exit("pyreadstat is required to read the SPSS file: pip install pyreadstat")

    frames, labels = [], {}
    for i, path in enumerate(paths):
        part, meta = pyreadstat.read_sav(str(path))
        part["_country"] = Path(path).stem
        part["_zone"] = part["JKZONE"].astype(str) + "_" + str(i)
        labels.update(meta.column_names_to_labels)
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)

    items = [
        c for c in df.columns
        if ITEM_RE.match(c) and df[c].notna().any()
        and np.allclose(df[c].dropna().values, np.round(df[c].dropna().values))
    ]

    # Group items by the booklets they appear in. Each such group is an item
    # block, and every student taking any of those booklets saw all of it.
    blocks = {}
    for c in items:
        blocks.setdefault(_block_key(df, c), []).append(c)
    blocks = {k: v for k, v in blocks.items() if k and len(v) >= 3}
    if not blocks:
        raise SystemExit("No item block could be identified from the booklet design.")

    return {"df": df, "labels": labels, "blocks": blocks,
            "min_coverage": min_coverage}


def prepare_block(state, booklets):
    """Score one item block into a complete response matrix."""
    df, labels = state["df"], state["labels"]
    items = state["blocks"][booklets]
    min_coverage = state["min_coverage"]

    sub = df[df["IDBOOK"].isin(booklets)].copy()
    cov = sub[items].notna().mean()
    keep = cov[cov >= min_coverage].index.tolist()

    # Only items every country in the pool administered can be compared, so
    # require the coverage threshold within each country rather than overall.
    per_country = []
    for _, g in sub.groupby("_country"):
        c = g[keep].notna().mean() if keep else pd.Series(dtype=float)
        per_country.append(set(c[c >= min_coverage].index))
    if per_country:
        keep = sorted(set(keep).intersection(*per_country))

    scored = {}
    for c in keep:
        s = _score_item(sub[c], labels.get(c, ""))
        if s is not None and s.notna().mean() >= min_coverage and 0 < s.mean() < 1:
            scored[c] = s

    resp = pd.DataFrame(scored).dropna()
    if resp.empty or resp.shape[1] < 3:
        return None
    sub = sub.loc[resp.index]

    return {
        "responses": resp.astype(int),
        "group": sub["ITSEX"].to_numpy(),          # 1 = girl, 2 = boy in TIMSS
        "weights": sub["TOTWGT"].to_numpy(float),
        "jkzone": sub["_zone"].to_numpy(),
        "country": sub["_country"].to_numpy(),
        "jkrep": sub["JKREP"].to_numpy(),
        "pv": sub[[f"BSMMAT{i:02d}" for i in range(1, 6)]].to_numpy(float).T,
        "booklets": booklets,
        "n_students": len(sub),
        "n_countries": sub["_country"].nunique(),
    }


def jk2_replicates(weights, jkzone, jkrep):
    """Build TIMSS JK2 replicate weights from the published design variables.

    Under JK2 each zone holds two primary sampling units, distinguished by
    ``JKREP``. Replicate ``h`` zeroes the weight of the unit with
    ``JKREP == 1`` in zone ``h`` and doubles its partner, leaving all other
    zones untouched. This is the scheme the TIMSS technical documentation
    prescribes, and reconstructing it from the released variables is what lets
    the design be reproduced without the assessment shipping the replicate
    weights themselves.
    """
    w = np.asarray(weights, dtype=float)
    z = np.asarray(jkzone)
    r = np.asarray(jkrep)
    zones = np.unique(z[~pd.isna(z)])

    out = np.tile(w, (len(zones), 1))
    for i, zone in enumerate(zones):
        in_zone = z == zone
        out[i, in_zone & (r == 1)] = 0.0
        out[i, in_zone & (r == 0)] *= 2.0
    return out, zones


def analyse_block(d, statistic, method="jackknife", fpc=None):
    """Run the full design-based comparison on one prepared block."""
    resp, grp = d["responses"], d["group"]
    w = normalize_weights(d["weights"])
    reps, zones = jk2_replicates(w, d["jkzone"], d["jkrep"])

    unw = mantel_haenszel(resp, grp, focal_label=1.0)
    wtd = mantel_haenszel(resp, grp, focal_label=1.0, weights=w)
    design = survey_dif(resp, grp, 1.0, w, replicate_weights=reps,
                        method=method, fpc=fpc, statistic=statistic)
    pv = survey_dif(resp, grp, 1.0, w, replicate_weights=reps,
                    plausible_values=d["pv"], method=method, fpc=fpc,
                    statistic=statistic)

    naive_se = (unw.se_log_alpha * 2.35).to_numpy()
    return pd.DataFrame({
        "block": "+".join(str(int(b)) for b in d["booklets"]),
        "n_students": d["n_students"],
        "n_replicates": reps.shape[0],
        "item": design.item,
        "delta_unweighted": unw.delta_mh,
        "delta_weighted": wtd.delta_mh,
        "se_naive": naive_se,
        "se_design": design.se,
        "se_ratio": design.se.to_numpy() / naive_se,
        "ci_low": design.ci_low,
        "ci_high": design.ci_high,
        "se_with_pv": pv.se,
        "fmi_with_pv": pv.fmi,
        "ets_class": wtd.ets_class,
    })


def main(sav, statistic="delta_mh", min_coverage=0.9, method="jackknife",
         fpc=None, max_blocks=None):
    OUT.mkdir(exist_ok=True)
    state = load_timss(sav, min_coverage=min_coverage)
    n_countries = state["df"]["_country"].nunique()

    order = sorted(state["blocks"], key=lambda k: -len(state["blocks"][k]))
    if max_blocks:
        order = order[: int(max_blocks)]

    print(f"TIMSS 2019 grade 8, {n_countries} country file(s), "
          f"{len(state['blocks'])} item blocks identified")

    parts, skipped = [], 0
    for booklets in order:
        d = prepare_block(state, booklets)
        if d is None:
            skipped += 1
            continue
        res = analyse_block(d, statistic, method=method, fpc=fpc)
        parts.append(res)
        print(f"  block {res.block.iloc[0]:<7} {d['n_students']:>5} students, "
              f"{len(res):>2} items, {res.n_replicates.iloc[0]:>3} replicates, "
              f"median SE ratio {res.se_ratio.median():.2f}")

    if not parts:
        raise SystemExit(
            "No block yielded a usable response matrix. TIMSS rotates blocks "
            "across booklets and countries administer different sets, so "
            "pooling many countries can empty every intersection. Try fewer "
            "countries, or lower --min-coverage."
        )

    summary = pd.concat(parts, ignore_index=True)
    ratio = summary.se_ratio.dropna()

    # Items appearing in more than one block yield independent estimates that
    # can be combined by inverse-variance weighting.
    pooled = []
    for item, g in summary.groupby("item"):
        if len(g) < 2:
            continue
        p = pool_estimates(g.delta_weighted, g.se_design, method="random")
        pooled.append({"item": item, "n_blocks": len(g), **p})
    if pooled:
        pooled = pd.DataFrame(pooled)
        pooled.to_csv(OUT / "timss_pooled_items.csv", index=False)

    print(f"\nPooled across {summary.block.nunique()} blocks "
          f"({skipped} skipped): {len(summary)} item analyses, "
          f"{summary.n_students.sum()} student-block observations")
    print(f"  Weighting shifts delta by a median of "
          f"{(summary.delta_weighted - summary.delta_unweighted).abs().median():.3f}")
    print(f"  Design-based SE / naive SE: median {ratio.median():.2f}, "
          f"IQR {ratio.quantile(.25):.2f}-{ratio.quantile(.75):.2f}")
    print(f"  With 5 plausible values: median FMI "
          f"{summary.fmi_with_pv.median():.3f}")

    if isinstance(pooled, pd.DataFrame) and len(pooled):
        print(f"  {len(pooled)} items appear in more than one block; "
              f"pooled median I-squared {pooled.i_squared.median():.3f}")
    else:
        print("  No item appears in more than one block, so no pooling applies")

    flagged = summary[summary.ets_class.isin(["B", "C"])]
    print(f"  {len(flagged)} of {len(summary)} item analyses at ETS class B or C by sex")
    excl = flagged[(flagged.ci_low > 0) | (flagged.ci_high < 0)]
    print(f"  of which {len(excl)} have a design-based interval excluding zero")

    summary.to_csv(OUT / "timss_survey_validation.csv", index=False)
    print(f"\nresults -> {OUT / 'timss_survey_validation.csv'}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sav", required=True, nargs="+",
                    help="one or more TIMSS student achievement .sav files")
    ap.add_argument("--statistic", default="delta_mh")
    ap.add_argument("--min-coverage", type=float, default=0.9, dest="min_coverage")
    ap.add_argument("--method", default="jackknife",
                    choices=["jackknife", "brr", "fay"])
    ap.add_argument("--fpc", type=float, default=None,
                    help="sampling fraction of primary sampling units")
    ap.add_argument("--max-blocks", type=int, default=None, dest="max_blocks")
    main(**vars(ap.parse_args()))
