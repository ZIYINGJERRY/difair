"""Real-data validation for DIFair (validation layer 4).

Two datasets, chosen because both are public and each exercises a different
half of the package.

Part A -- Verbal aggression data (Vansteelandt, 2000), 316 respondents and 24
dichotomous items, distributed with difR and used as the worked example in the
difR paper. Item-level procedures are compared against difR on real response
patterns rather than simulated ones.

Part B -- Open University Learning Analytics Dataset (Kuzilek et al., 2017),
32,593 students with demographics and assessment scores. Assessments within a
cohort are dichotomised at the pass mark to form an item matrix, which lets the
item-level, model-level and attribution layers all run on the same real
pipeline: assessment -> score -> at-risk prediction -> decision.

Usage
-----
    python examples/realdata_validation.py --difr-dir /tmp/difR \\
        --oulad-dir /path/to/oulad
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from difair.dif import breslow_day, detect_dif, logistic_dif, mantel_haenszel, standardization
from difair.fairness import fairness_report
from difair.pipeline import attribute_stages

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "validation_results"

R_TEMPLATE = """
for (f in c("mantelHaenszel.R","stdPDIF.R","breslowDay.R","Logistik.R"))
  suppressWarnings(source(file.path("{rdir}", f)))
d <- read.csv("{csv}")
member <- d$member
data <- as.matrix(d[, setdiff(names(d), "member")])
mh <- mantelHaenszel(data, member, correct = TRUE)
st <- stdPDIF(data, member)
bd <- breslowDay(data, member, BDstat = "BD")
lg <- Logistik(data, member, type = "both", criterion = "LRT")
write.csv(data.frame(
  item = colnames(data),
  difR_mh_chi2 = mh$resMH, difR_alpha_mh = mh$resAlpha,
  difR_var_lambda = mh$varLambda, difR_std_pdif = st$resStd,
  difR_bd_stat = bd$res[,1], difR_logistik_chi2 = lg$stat,
  difR_logistik_delta_r2 = lg$deltaR2
), "{out}", row.names = FALSE)
"""


def _agree(label, r, p, tol):
    sub = pd.DataFrame({"r": r, "p": p}).dropna()
    if sub.empty:
        return {"quantity": label, "n": 0, "max_abs_diff": np.nan, "agrees": False}
    a = (sub.r - sub.p).abs()
    rel = a / sub.r.abs().clip(lower=1e-12)
    return {
        "quantity": label,
        "n": int(len(sub)),
        "max_abs_diff": float(a.max()),
        "max_rel_diff": float(rel.max()),
        "agrees": bool(rel.max() < tol or a.max() < tol),
    }


# --------------------------------------------------------------------------- #
def part_a_verbal(difr_dir, rscript="Rscript"):
    """Item-level agreement on the verbal aggression data."""
    raw = pd.read_csv(Path(difr_dir) / "data" / "verbal.txt", sep="\t")
    items = [c for c in raw.columns if c not in ("Anger", "Gender")]
    df = raw[items].astype(int).copy()
    df["member"] = raw["Gender"].astype(int)      # difR convention: 1 = focal

    n_focal = int(df.member.sum())
    print(f"\nPart A: verbal aggression data, {len(df)} respondents, "
          f"{len(items)} items, {n_focal} focal / {len(df) - n_focal} reference")

    tmp_csv = OUT / "_verbal.csv"
    tmp_out = OUT / "realdata_verbal_R.csv"
    df.to_csv(tmp_csv, index=False)

    script = R_TEMPLATE.format(rdir=str(Path(difr_dir) / "R"), csv=tmp_csv, out=tmp_out)
    (OUT / "_verbal.R").write_text(script)
    subprocess.run([rscript, str(OUT / "_verbal.R")], check=True, capture_output=True)
    ref = pd.read_csv(tmp_out)

    resp = df.drop(columns="member")
    mem = df.member.to_numpy()
    mh = mantel_haenszel(resp, mem, focal_label=1, clamp_correction=False)
    st = standardization(resp, mem, focal_label=1)
    bd = breslow_day(resp, mem, focal_label=1, tarone=False)
    lg = logistic_dif(resp, mem, focal_label=1, standardize_matching=False)

    rows = [
        _agree("MH chi-square", ref.difR_mh_chi2, mh.chi2, 1e-6),
        _agree("MH common odds ratio", ref.difR_alpha_mh, mh.alpha_mh, 1e-8),
        _agree("MH var(log alpha)", ref.difR_var_lambda, mh.se_log_alpha ** 2, 1e-8),
        _agree("Standardized P-DIF", ref.difR_std_pdif, st.std_p_dif, 1e-8),
        _agree("Breslow-Day statistic", ref.difR_bd_stat, bd.bd_stat, 1e-4),
        _agree("Logistic LRT chi-square", ref.difR_logistik_chi2, lg.chi2_total, 1e-5),
        _agree("Logistic Nagelkerke dR2", ref.difR_logistik_delta_r2, lg.delta_r2, 1e-5),
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "realdata_verbal_agreement.csv", index=False)
    print(summary.to_string(index=False))

    flagged = mantel_haenszel(resp, mem, focal_label=1)
    n_bc = int(flagged.ets_class.isin(["B", "C"]).sum())
    print(f"  DIFair flags {n_bc} of {len(items)} items at ETS class B or C")
    for _, r in flagged.nlargest(3, "delta_mh", keep="all").head(3).iterrows():
        print(f"    {r['item']:<16} delta = {r['delta_mh']:+.3f}  class {r['ets_class']}")
    return summary


# --------------------------------------------------------------------------- #
def part_b_oulad(oulad_dir, min_coverage=0.5, pass_mark=40, min_students=800):
    """End-to-end audit of a real learning-analytics pipeline."""
    d = Path(oulad_dir)
    si = pd.read_csv(d / "studentInfo.csv")
    sa = pd.read_csv(d / "studentAssessment.csv")

    # Choose the cohort offering the most widely-taken assessments, subject to
    # a minimum enrolment: item-level procedures need a reasonable test length.
    best, cohort, keep = -1, None, []
    for (mod, pres), g in si.groupby(["code_module", "code_presentation"]):
        if len(g) < min_students:
            continue
        s = sa[sa.id_student.isin(set(g.id_student))]
        cov = s.groupby("id_assessment").id_student.nunique() / len(g)
        k = cov[cov >= min_coverage].index.tolist()
        if len(k) > best:
            best, cohort, keep = len(k), (mod, pres), k
    if cohort is None:
        raise ValueError("No cohort met the enrolment threshold.")

    coh = si[(si.code_module == cohort[0]) & (si.code_presentation == cohort[1])].copy()
    sub = sa[sa.id_student.isin(coh.id_student) & sa.id_assessment.isin(keep)]

    wide = (sub.pivot_table(index="id_student", columns="id_assessment",
                            values="score", aggfunc="max"))
    # Reindex to the full cohort: a student who never submitted an assessment
    # did not pass it, which is the substantively correct reading and keeps the
    # withdrawn students -- precisely the at-risk group -- in the analysis.
    wide = wide.reindex(coh.id_student.values)
    coh = coh.set_index("id_student").loc[wide.index]

    # Dichotomise each assessment at the pass mark to form an item matrix.
    resp = (wide.fillna(0) >= pass_mark).astype(int)
    resp.columns = [f"assess_{c}" for c in resp.columns]

    print(f"\nPart B: OULAD cohort {cohort[0]}-{cohort[1]}, {len(resp)} students, "
          f"{resp.shape[1]} assessments with >= {min_coverage:.0%} coverage")

    results = {}
    for attr, focal in (("disability", "Y"), ("gender", "F")):
        grp = coh[attr].to_numpy()
        if resp.shape[1] < 3 or len(np.unique(grp)) != 2:
            continue
        dif = detect_dif(resp, grp, focal_label=focal, methods=("mh", "std"), purify=True)
        n_bc = len(dif.flagged)
        print(f"  DIF by {attr} (focal = {focal}): {n_bc} of {resp.shape[1]} "
              f"assessments at ETS B/C")
        results[f"dif_{attr}"] = dif.table.assign(attribute=attr)

    # Model level: predict "not passing" from the assessment record.
    at_risk = coh.final_result.isin(["Fail", "Withdrawn"]).astype(int).to_numpy()
    grp = coh.disability.to_numpy()
    score = resp.to_numpy().sum(axis=1)
    cut = np.quantile(score, 0.35)
    pred = (score <= cut).astype(int)          # flag the weakest 35 percent
    fair = fairness_report(at_risk, pred, grp, "Y")
    print("  Model-level fairness of the at-risk flag (focal = disabled):")
    for _, r in fair.iterrows():
        print(f"    {r['metric']:<22} {r['value']:+.4f}")
    results["fairness"] = fair.drop(columns=["detail"])

    # Attribution: a genuine proxy is prior attempts, which correlates with
    # disadvantage and is routinely available to an institutional model.
    proxy = coh.num_of_prev_attempts.to_numpy(dtype=float)
    rng = np.random.default_rng(0)
    train_mask = rng.random(len(coh)) < 0.7
    dif_items = detect_dif(resp, grp, focal_label="Y",
                           methods=("mh",), purify=True).flagged
    att = attribute_stages(
        resp, grp, "Y", at_risk, dif_items=dif_items, proxy=proxy,
        train_mask=train_mask, selection_rate=0.35, seed=0,
    )
    print(f"  Attribution: baseline gap {att.baseline_gap:.4f} -> "
          f"residual {att.residual_gap:.4f}")
    print(att.summary().round(4).to_string(index=False))
    results["attribution"] = att.summary()

    for k, v in results.items():
        v.to_csv(OUT / f"realdata_oulad_{k}.csv", index=False)
    return results


# --------------------------------------------------------------------------- #
def main(difr_dir=None, oulad_dir=None, rscript="Rscript"):
    OUT.mkdir(exist_ok=True)
    if difr_dir:
        part_a_verbal(difr_dir, rscript)
    if oulad_dir:
        part_b_oulad(oulad_dir)
    if not (difr_dir or oulad_dir):
        sys.exit("Supply --difr-dir and/or --oulad-dir.")
    print(f"\nresults -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--difr-dir", help="difR repository root (contains R/ and data/)")
    ap.add_argument("--oulad-dir", help="directory with studentInfo.csv and studentAssessment.csv")
    ap.add_argument("--rscript", default="Rscript")
    main(**vars(ap.parse_args()))
