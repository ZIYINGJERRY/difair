"""End-to-end DIFair walkthrough: items, model, attribution, report."""

from difair.dif import detect_dif
from difair.fairness import fairness_report
from difair.pipeline import attribute_stages
from difair.report import audit_report
from difair.simulate import simulate_pipeline_data

d = simulate_pipeline_data(n_ref=1500, n_focal=1500, seed=1)

# 1. item level -------------------------------------------------------------
dif = detect_dif(d["responses"], d["group"], focal_label="focal", purify=True)
print(f"Flagged {len(dif.flagged)} of {len(dif.table)} items: {dif.flagged}")
print(dif.summary(), "\n")

# 2. model level ------------------------------------------------------------
score = d["responses"].to_numpy().sum(axis=1)
pred = (score >= sorted(score)[int(0.6 * len(score))]).astype(int)
fair = fairness_report(d["outcome"], pred, d["group"], "focal")
print(fair[["metric", "value"]].to_string(index=False), "\n")

# 3. stage attribution ------------------------------------------------------
att = attribute_stages(
    d["responses"], d["group"], "focal", d["outcome"],
    dif_items=dif.flagged, proxy=d["proxy"],
    train_mask=d["train_mask"], train_outcome=d["outcome_observed"],
)
print(f"baseline {att.baseline_gap:.4f} -> residual {att.residual_gap:.4f}")
print(att.summary().to_string(index=False), "\n")

# 4. report -----------------------------------------------------------------
path = audit_report(
    "difair_audit.html", dif_result=dif, fairness_table=fair, attribution=att,
    context={"Instrument": "Simulated 30-item test", "Focal group": "focal"},
)
print("report written to", path)
