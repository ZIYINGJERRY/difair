"""Cross-validation of DIFair against the difR reference implementation.

Layer 1 of the validation plan. Generates a range of datasets, has difR's own R
sources and base R's ``stats::mantelhaen.test`` compute the reference values,
then checks that DIFair reproduces them numerically.

Both sides must agree to tight tolerance on quantities that are mathematically
determined; where the two libraries make different but defensible choices, the
difference is reported rather than hidden.

Usage
-----
    Rscript examples/crossvalidate_difR.R /path/to/difR/R  xval_data  xval_R.csv
    python examples/crossvalidate_difR.py

or simply::

    python examples/crossvalidate_difR.py --difr-dir /path/to/difR/R

which drives the R step for you.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from difair.dif import breslow_day, logistic_dif, mantel_haenszel, standardization
from difair.simulate import simulate_dif_data

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "xval_data"
OUT = ROOT / "validation_results"

# Datasets spanning small/large N, short/long tests, uniform and non-uniform
# DIF, group impact, and a null case with no DIF at all.
CASES = {
    "small_uniform": dict(n_ref=300, n_focal=300, n_items=10, n_dif_items=3,
                          dif_magnitude=0.8, seed=101),
    "large_uniform": dict(n_ref=2000, n_focal=2000, n_items=25, n_dif_items=5,
                          dif_magnitude=0.6, seed=102),
    "nonuniform": dict(n_ref=1200, n_focal=1200, n_items=15, n_dif_items=4,
                       dif_magnitude=0.3, nonuniform_magnitude=0.8, seed=103),
    "with_impact": dict(n_ref=1000, n_focal=1000, n_items=20, n_dif_items=4,
                        dif_magnitude=0.7, impact=0.8, seed=104),
    "null_no_dif": dict(n_ref=1500, n_focal=1500, n_items=20, n_dif_items=0,
                        dif_magnitude=0.0, seed=105),
    "unbalanced_groups": dict(n_ref=2500, n_focal=500, n_items=18, n_dif_items=4,
                              dif_magnitude=0.9, seed=106),
}


def write_datasets():
    DATA.mkdir(exist_ok=True)
    for name, kw in CASES.items():
        sim = simulate_dif_data(**kw)
        df = sim.responses.copy()
        df["member"] = (sim.group == "focal").astype(int)  # difR: 1 = focal
        df.to_csv(DATA / f"dataset_{name}.csv", index=False)
    print(f"wrote {len(CASES)} datasets to {DATA}")


def python_side():
    rows = []
    for name in CASES:
        df = pd.read_csv(DATA / f"dataset_{name}.csv")
        member = df.pop("member").to_numpy()

        # clamp_correction=False reproduces difR / stats::mantelhaen.test exactly
        mh = mantel_haenszel(df, member, focal_label=1, correct=True,
                             clamp_correction=False)
        st = standardization(df, member, focal_label=1)
        # difR applies no Tarone correction and reports df = n_strata - 1
        bd = breslow_day(df, member, focal_label=1, tarone=False)
        lg = logistic_dif(df, member, focal_label=1, standardize_matching=False)

        rows.append(pd.DataFrame({
            "dataset": name,
            "item": mh.item,
            "py_mh_chi2": mh.chi2.values,
            "py_alpha_mh": mh.alpha_mh.values,
            "py_var_lambda": mh.se_log_alpha.values ** 2,
            "py_std_pdif": st.std_p_dif.values,
            "py_bd_stat": bd.bd_stat.values,
            "py_logistik_chi2": lg.chi2_total.values,
            "py_logistik_delta_r2": lg.delta_r2.values,
        }))
    return pd.concat(rows, ignore_index=True)


def compare(merged):
    """Compare each quantity and summarise agreement."""
    pairs = [
        ("MH chi-square", "difR_mh_chi2", "py_mh_chi2", 1e-6),
        ("MH chi-square (base R)", "baseR_mh_chi2", "py_mh_chi2", 1e-6),
        ("MH common odds ratio", "difR_alpha_mh", "py_alpha_mh", 1e-8),
        ("MH var(log alpha), RBG", "difR_var_lambda", "py_var_lambda", 1e-8),
        ("Standardized P-DIF", "difR_std_pdif", "py_std_pdif", 1e-8),
        # difR rounds the Breslow-Day statistic to four decimals on output.
        ("Breslow-Day statistic", "difR_bd_stat", "py_bd_stat", 1e-4),
        ("Logistic LRT chi-square", "difR_logistik_chi2", "py_logistik_chi2", 1e-5),
        ("Logistic Nagelkerke dR2", "difR_logistik_delta_r2", "py_logistik_delta_r2", 1e-5),
    ]
    out = []
    for label, rcol, pcol, tol in pairs:
        if rcol not in merged.columns:
            continue
        sub = merged[[rcol, pcol]].dropna()
        if sub.empty:
            out.append({"quantity": label, "n": 0, "max_abs_diff": np.nan,
                        "max_rel_diff": np.nan, "tolerance": tol, "agrees": False})
            continue
        r, p = sub[rcol].to_numpy(float), sub[pcol].to_numpy(float)
        abs_d = np.abs(r - p)
        denom = np.maximum(np.abs(r), 1e-12)
        rel_d = abs_d / denom
        out.append({
            "quantity": label,
            "n": int(len(sub)),
            "max_abs_diff": float(abs_d.max()),
            "max_rel_diff": float(rel_d.max()),
            "tolerance": tol,
            "agrees": bool(rel_d.max() < tol or abs_d.max() < tol),
        })
    return pd.DataFrame(out)


def main(difr_dir=None, rscript="Rscript"):
    OUT.mkdir(exist_ok=True)
    write_datasets()

    r_csv = OUT / "crossvalidation_R.csv"
    if difr_dir:
        if shutil.which(rscript) is None:
            sys.exit(f"{rscript} not found on PATH.")
        print("running R reference implementation")
        subprocess.run(
            [rscript, str(Path(__file__).with_suffix(".R")), difr_dir, str(DATA), str(r_csv)],
            check=True,
        )
    if not r_csv.exists():
        sys.exit(f"{r_csv} not found; run the R script first or pass --difr-dir.")

    r_side = pd.read_csv(r_csv)
    py = python_side()
    merged = r_side.merge(py, on=["dataset", "item"], how="inner")
    merged.to_csv(OUT / "crossvalidation_merged.csv", index=False)

    summary = compare(merged)
    summary.to_csv(OUT / "crossvalidation_summary.csv", index=False)

    print(f"\ncompared {len(merged)} item-level statistics "
          f"across {merged.dataset.nunique()} datasets\n")
    with pd.option_context("display.width", 120):
        print(summary.to_string(index=False))

    failed = summary[~summary.agrees]
    if len(failed):
        print("\nQuantities outside tolerance:")
        print(failed.to_string(index=False))
    else:
        print("\nAll quantities agree within tolerance.")
    return int(len(failed))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--difr-dir", help="path to difR's R/ source directory")
    ap.add_argument("--rscript", default="Rscript")
    a = ap.parse_args()
    sys.exit(main(a.difr_dir, a.rscript))
