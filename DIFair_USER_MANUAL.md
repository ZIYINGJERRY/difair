# DIFair User Manual

**Version 0.7.0** · Unified differential item functioning (DIF) and algorithmic fairness auditing for educational data

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Requirements](#2-system-requirements)
3. [Installation](#3-installation)
4. [Functional Modules](#4-functional-modules)
5. [API Reference](#5-api-reference)
6. [Operation Guide](#6-operation-guide)
7. [DIFair Studio (Browser Tool)](#7-difair-studio-browser-tool)
8. [Validation and Accuracy](#8-validation-and-accuracy)
9. [Troubleshooting](#9-troubleshooting)
10. [Support and Version Information](#10-support-and-version-information)
11. [Appendix](#11-appendix)

---

## 1. Introduction

### 1.1 Overview

DIFair audits whether an assessment — and the decision pipeline built on top of it — treats demographic groups fairly. It brings together two research traditions that have almost no software in common:

- **Differential item functioning (DIF)** from educational measurement, historically implemented in R (`difR`, `mirt`, `difNLR`, `dexter`)
- **Algorithmic fairness metrics** from machine learning, historically implemented in Python (`fairlearn`, `AIF360`, `aequitas`)

Researchers auditing a modern educational pipeline need both, and currently move data between languages to get them. DIFair provides both behind one Python API, and adds a capability neither tradition offers: **given a group gap in the decisions a system makes, which stage of the pipeline produced it?**

### 1.2 Key Features

- **Item-level DIF** — Mantel–Haenszel with the ETS delta scale, Swaminathan–Rogers logistic regression, Dorans–Kulick standardization, Breslow–Day with the Tarone correction, and iterative matching-score purification
- **Polytomous DIF** — generalized Mantel–Haenszel with the Zwick–Thayer SMD, and proportional-odds ordinal logistic regression separating uniform from non-uniform DIF
- **Model-level fairness** — demographic parity, equalized odds, equal opportunity, predictive parity, calibration gap, and ordinal disparity for graded decisions
- **Design-based inference** — replicate-weight variance (jackknife, BRR, Fay), plausible-value combination by Rubin's rules, automatic detection of how replicate weights were constructed, and inverse-variance pooling across analyses
- **Pipeline attribution** — exact Shapley decomposition of a decision disparity onto the item, sampling and model stages
- **Simulation** — 2PL and generalized partial credit response generation with planted DIF and known ground truth
- **Reporting** — a standalone, self-contained HTML audit report
- **Zero-install browser front end** — `difair_studio.html`, a single file that runs the same analyses locally in a browser with no Python, no installation and no network access

### 1.3 Technical Architecture

| Layer | Technology |
|---|---|
| Language | Python ≥ 3.9 |
| Numerical core | NumPy, SciPy |
| Data handling | pandas |
| Statistical models | statsmodels, scikit-learn |
| Browser front end | Single-file HTML + vanilla JavaScript (no framework, no CDN, no network calls) |
| Packaging | setuptools / PEP 621 (`pyproject.toml`) |
| Continuous integration | GitHub Actions — Ubuntu, macOS, Windows × Python 3.9, 3.11, 3.12 |
| License | MIT |

### 1.4 Two Ways to Run an Audit

|  | Intended for | How to run |
|---|---|---|
| **`difair_studio.html`** | Anyone; no programming required | Open the file in a browser. No install, no Python, no network calls. |
| **`difair` Python package** | Python users; research-grade reproducibility | `pip install git+https://github.com/ZIYINGJERRY/difair` |

`difair_studio.html` ships alongside the package in the repository rather than inside the installable Python distribution. Use the Python package when you need bit-exact reproducibility with the validated reference implementation, an HTML audit report bundling everything together, or scripted analysis across many files.

---

## 2. System Requirements

### 2.1 Python Package

| Item | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.12 |
| Operating system | Linux, macOS, Windows | any of the three (all CI-tested) |
| Memory | 2 GB | 8 GB for large-scale assessment files |
| Disk | 100 MB | 1 GB including validation datasets |

**Required dependencies** (installed automatically):

| Package | Minimum version |
|---|---|
| numpy | 1.22 |
| pandas | 1.5 |
| scipy | 1.9 |
| scikit-learn | 1.1 |
| statsmodels | 0.13 |

**Optional dependency groups:**

| Group | Packages | Needed for |
|---|---|---|
| `dev` | pytest ≥ 7, pytest-cov | running the test suite |
| `timss` | pyreadstat ≥ 1.2 | reading TIMSS SPSS files in `examples/timss_validation.py` |

Cross-validation against `difR` (`examples/crossvalidate_difR.py`) additionally requires **base R** on the system path. No R packages need to be installed — difR's core computational functions depend only on `stats` and are sourced directly.

### 2.2 Browser Tool

| Item | Requirement |
|---|---|
| Browser | Chrome 80+, Firefox 75+, Edge 80+, Safari 13+ |
| Resolution | 1280 × 720 minimum |
| Network | **Not required.** The page performs no network requests and works fully offline. |
| Installation | None. Open the local file directly. |

Because nothing is uploaded anywhere, the browser tool is safe to use with confidential student or respondent data.

---

## 3. Installation

### 3.1 From GitHub

DIFair is not on PyPI. Install it directly from the repository:

```bash
pip install git+https://github.com/ZIYINGJERRY/difair
```

### 3.2 From Source

Clone instead if you also want `difair_studio.html`, the examples or the
validation data, none of which are part of the installable package:

```bash
git clone https://github.com/ZIYINGJERRY/difair
cd difair
pip install -e ".[dev]"
```

### 3.3 Verify the Installation

```bash
python -c "import difair; print(difair.__version__)"
# 0.7.0

pytest tests/ -q --cov=difair
python examples/quickstart.py
```

`examples/quickstart.py` runs the full item → model → attribution → report chain on simulated data and writes `difair_audit.html`. If that file is produced, the installation is complete.

### 3.4 Browser Tool

No installation. Download `difair_studio.html` from the repository and double-click it, or open it from the browser's File menu.

---

## 4. Functional Modules

| Module | Purpose |
|---|---|
| `difair.dif` | Dichotomous DIF: Mantel–Haenszel, logistic regression, standardization, Breslow–Day, iterative purification, survey-weight normalization |
| `difair.poly` | Polytomous DIF: generalized Mantel–Haenszel, proportional-odds logistic regression, purification for ordered items |
| `difair.survey` | Design-based inference: replicate-weight variance, plausible values, replicate-design detection, cross-analysis pooling |
| `difair.fairness` | Model-level fairness metrics, including ordinal disparity for graded decisions |
| `difair.pipeline` | Exact Shapley attribution of a decision disparity to pipeline stages |
| `difair.simulate` | 2PL and generalized partial credit simulation with planted DIF |
| `difair.report` | Standalone HTML audit report |

### 4.1 `difair.dif` — Item-Level DIF

Detects whether respondents of equal ability but different group membership have unequal probabilities of answering an item correctly.

- **`mantel_haenszel`** — the ETS operational procedure. Reports the MH chi-square, common odds ratio, the ETS delta (`delta_mh`), a p-value, and an A/B/C classification. Supports survey weights and an external matching criterion.
- **`logistic_dif`** — fits three nested models per item and separates *uniform* DIF (a constant group advantage) from *non-uniform* DIF (an advantage that varies with ability), reporting likelihood-ratio tests and the Zumbo–Thomas Nagelkerke effect size.
- **`standardization`** — Dorans–Kulick STD P-DIF. Conventional bands: `|STD P-DIF| < 0.05` negligible, `0.05–0.10` moderate, `≥ 0.10` large.
- **`breslow_day`** — tests whether the odds ratio is homogeneous across matching strata; a significant result indicates non-uniform DIF. The Tarone correction is applied by default.
- **`purify_matching_score`** — iteratively removes flagged items from the matching total. An item carrying DIF contaminates the yardstick applied to every other item; purification removes that contamination.
- **`normalize_weights`** — rescales population-scale survey weights to sum to the sample size, returning chi-square statistics to their nominal scale.
- **`detect_dif`** — runs several procedures and merges them into one table per item. This is the normal entry point.

### 4.2 `difair.poly` — Polytomous DIF

For rating scales, Likert questionnaires and partial-credit items.

- **`generalized_mantel_haenszel`** — the Mantel chi-square for ordered categories with the Zwick–Thayer standardized mean difference as effect size. Negative SMD indicates DIF against the focal group.
- **`ordinal_logistic_dif`** — nested cumulative-logit (proportional-odds) models separating uniform from non-uniform DIF.
- **`purify_poly_matching`** — the polytomous counterpart of matching-score purification, classified by SMD band.
- **`detect_dif_poly`** — merged entry point. Returns the same `DIFResult` container as `detect_dif`, so downstream code and `audit_report` accept either.

### 4.3 `difair.survey` — Design-Based Inference

Large-scale assessments sample with unequal probability and in clusters. Naive standard errors computed as if the sample were simple random are too small, often by an order of magnitude.

- **`jackknife_weights`** — constructs paired jackknife replicate weights from a stratum/PSU design when the assessment does not supply its own.
- **`replicate_variance`** — converts replicate spread into a sampling variance under one of three constants: `jackknife` (TIMSS, PIRLS), `brr` (NAEP), `fay` (PISA).
- **`infer_replicate_design`** — reads the construction off the weights themselves. This matters: applying Fay's constant to TIMSS JK2 replicates shrinks standard errors to roughly a sixth of their proper size, and nothing in the output signals the error. When no signature matches, the return value reports what was observed — zeroed, unchanged and doubled fractions, and the spread of non-zero multipliers — so an unusual design can be distinguished from bad input.
- **`combine_plausible_values`** — Rubin's rules across plausible values, reporting the fraction of missing information (`fmi`).
- **`survey_dif`** — the combined entry point: runs the DIF procedure once per plausible value and once per replicate, takes replicate variance within each plausible value, then combines across values.
- **`pool_estimates`** — inverse-variance pooling of independent estimates of the same item, with Cochran's Q and I², and a random-effects option.

### 4.4 `difair.fairness` — Model-Level Metrics

- **`group_rates`** — per-group confusion-matrix rates: `n`, `selection_rate`, `tpr`, `fpr`, `tnr`, `fnr`, `ppv`, `accuracy`
- **`demographic_parity`** — selection-rate difference and ratio
- **`equalized_odds`** — TPR and FPR gaps; the violation is the larger absolute gap
- **`equal_opportunity`** — the TPR gap alone
- **`predictive_parity`** — the positive predictive value gap
- **`calibration_gap`** — group-wise expected calibration error and its focal–reference gap
- **`fairness_report`** — runs every applicable metric into a tidy table, signed so that negative values indicate the focal group is disadvantaged
- **`ordinal_disparity`** — for graded decisions. Binary metrics collapse an ordered outcome to a single cut, treating a system that pushes the focal group from the top band to the second identically to one that pushes them to the bottom. Reports the mean difference, standardized difference, and the probability of superiority (0.5 under equal treatment) with a Hanley–McNeil standard error.
- **`ordinal_disparity_replicate`** — the same statistic with a design-based standard error, for clustered surveys where the Hanley–McNeil independence assumption fails.
- **`ordinal_group_summary`** — per-group distribution of an ordered outcome.

### 4.5 `difair.pipeline` — Stage Attribution

Treats each pipeline stage as a player in a cooperative game whose characteristic function is the disparity remaining after that subset of mitigations is applied. With three or four stages the Shapley values are computed **exactly**, not sampled.

Default stages: `("item", "sampling", "model")`. A `"decision"` stage is also available.

- **`attribute_stages`** — returns an `AttributionResult` with per-stage Shapley values, the baseline gap, the residual gap after all mitigations, and the full coalition-value dictionary.

### 4.6 `difair.simulate` — Simulation

- **`simulate_dif_data`** — 2PL responses with a known set of DIF items. Controls test length, group sizes, DIF magnitude, non-uniform magnitude, and *impact* (a genuine ability difference, which must not be mistaken for DIF).
- **`simulate_poly_dif_data`** — generalized partial credit responses with planted DIF.
- **`simulate_pipeline_data`** / **`simulate_poly_pipeline_data`** — end-to-end assessment-to-decision pipelines in which both groups are given **identical ability distributions and identical true outcome rates**, so every observed disparity is a planted artefact with a known stage of origin.

### 4.7 `difair.report` — Audit Report

- **`audit_report`** — writes a single self-contained HTML file combining the DIF table, the fairness table and the attribution result, with a free-form `context` dictionary rendered in the header so the report is self-describing.

---

## 5. API Reference

### 5.1 Item-Level DIF

```python
mantel_haenszel(responses, group, focal_label=None, matching=None,
                include_studied_item=True, correct=True, clamp_correction=True,
                weights=None, alpha=0.05)
```

| Parameter | Type | Description |
|---|---|---|
| `responses` | DataFrame `(n_persons, n_items)` | Dichotomous 0/1 responses; `NaN` marks missing |
| `group` | array-like `(n_persons,)` | Binary group membership |
| `focal_label` | hashable | Value of `group` identifying the focal group. **Required whenever labels are not 0/1** |
| `matching` | array-like, optional | External matching criterion; defaults to the observed total score |
| `include_studied_item` | bool | Whether the studied item contributes to the matching score. The operational ETS convention includes it |
| `correct` | bool | Apply the continuity correction |
| `clamp_correction` | bool | Floor the corrected statistic at zero. Set `False` to reproduce difR and `mantelhaen.test` exactly |
| `weights` | array-like, optional | Survey weights; normalize first |
| `alpha` | float | Significance level for the ETS classification |

Returns a DataFrame with `item`, `delta_mh`, `p_value`, `ets_class`, `n_strata`, `favors` and related columns.

```python
logistic_dif(responses, group, focal_label=None, matching=None,
             include_studied_item=True, standardize_matching=True)
```

Fits, per item, with `S` the matching score and `G` the focal indicator:

```
M0:  logit(p) = b0 + b1*S
M1:  logit(p) = b0 + b1*S + b2*G
M2:  logit(p) = b0 + b1*S + b2*G + b3*S*G
```

Returns `chi2_total`/`p_total` (2 df), `chi2_uniform`/`p_uniform` (1 df), `chi2_nonuniform`/`p_nonuniform` (1 df), `beta_group`, `beta_interaction`, `delta_r2`, `zt_class`.

```python
standardization(responses, group, focal_label=None, matching=None,
                include_studied_item=True, weights=None)

breslow_day(responses, group, focal_label=None, matching=None,
            include_studied_item=True, tarone=True, weights=None)

purify_matching_score(responses, group, focal_label=None, max_iter=5,
                      alpha=0.05, verbose=False)   # -> (matching, flagged)

normalize_weights(weights, group=None)

detect_dif(responses, group, focal_label=None,
           methods=("mh", "logistic", "std", "bd"), purify=False, alpha=0.05)
```

`detect_dif` returns a **`DIFResult`**:

| Attribute | Description |
|---|---|
| `.table` | Merged per-item statistics |
| `.flagged` | Items reaching ETS class B or C |
| `.focal_label` | The focal group label used |
| `.purified` | Whether purification was applied |
| `.summary()` | Counts by ETS class |

### 5.2 Polytomous DIF

```python
generalized_mantel_haenszel(responses, group, focal_label=None, matching=None,
                            include_studied_item=True, weights=None, alpha=0.05)

ordinal_logistic_dif(responses, group, focal_label=None, matching=None,
                     include_studied_item=True, standardize_matching=True)

purify_poly_matching(responses, group, focal_label=None, max_iter=5, alpha=0.05)

detect_dif_poly(responses, group, focal_label=None,
                methods=("gmh", "ordinal"), purify=False, weights=None, alpha=0.05)
```

### 5.3 Fairness Metrics

```python
group_rates(y_true, y_pred, group)
demographic_parity(y_pred, group, focal_label)
equalized_odds(y_true, y_pred, group, focal_label)
equal_opportunity(y_true, y_pred, group, focal_label)
predictive_parity(y_true, y_pred, group, focal_label)
calibration_gap(y_true, y_score, group, focal_label, n_bins=10)
fairness_report(y_true, y_pred, group, focal_label, y_score=None)

ordinal_group_summary(y_true, group)
ordinal_disparity(y_true, group, focal_label, n_levels=None)
ordinal_disparity_replicate(y_true, group, focal_label, weights=None,
                            replicate_weights=None, method="jackknife",
                            fay_factor=0.5, fpc=None)
```

`fairness_report` returns columns `metric`, `value`, `detail`. If `y_true` is `None`, only demographic parity is computed.

### 5.4 Design-Based Inference

```python
jackknife_weights(weights, strata, psu=None, n_replicates=None, seed=None)

replicate_variance(estimate, replicate_estimates, method="jackknife",
                   fay_factor=0.5, fpc=None)

infer_replicate_design(replicate_weights, base_weights=None, tol=0.5)

combine_plausible_values(estimates, variances=None)

survey_dif(responses, group, focal_label, weights, replicate_weights=None,
           plausible_values=None, statistic="delta_mh", n_matching_bins=20,
           method="jackknife", fay_factor=0.5, fpc=None, check_design=True,
           polytomous=False, alpha=0.05)

pool_estimates(estimates, standard_errors, method="fixed")
```

`survey_dif` returns a DataFrame with `item`, `estimate`, `se`, `ci_low`, `ci_high`, `fmi`.

`combine_plausible_values` returns `estimate`, `variance`, `se`, `within`, `between`, `df`, `fmi`, `n_values`.

### 5.5 Pipeline Attribution

```python
DEFAULT_STAGES = ("item", "sampling", "model")

attribute_stages(responses, group, focal_label, outcome, dif_items,
                 proxy=None, train_mask=None, train_outcome=None,
                 stages=DEFAULT_STAGES, metric="demographic_parity",
                 selection_rate=0.4, seed=0)
```

Returns an **`AttributionResult`**: `.shapley`, `.baseline_gap`, `.residual_gap`, `.metric`, `.coalition_values`, `.response_kind`, `.explained()`, `.summary()`.

Dichotomous and ordered categorical responses are both accepted (the stage score is the item sum in either case). Continuous responses are rejected.

### 5.6 Simulation and Reporting

```python
simulate_dif_data(n_ref=1000, n_focal=1000, n_items=30, n_dif_items=5,
                  dif_magnitude=0.6, nonuniform_magnitude=0.0, impact=0.0,
                  a_range=(0.7, 1.8), b_range=(-2.0, 2.0), seed=None)

simulate_poly_dif_data(n_ref=1000, n_focal=1000, n_items=20, n_categories=5,
                       n_dif_items=4, dif_magnitude=0.6, impact=0.0,
                       a_range=(0.7, 1.8), seed=None)

simulate_pipeline_data(n_ref=1500, n_focal=1500, n_items=30, n_dif_items=6,
                       dif_magnitude=0.8, proxy_strength=0.9, label_bias=0.35,
                       undersample_focal=0.55, outcome_noise=0.5, seed=None)

simulate_poly_pipeline_data(...)   # polytomous counterpart

audit_report(path, dif_result=None, fairness_table=None, attribution=None,
             title="DIFair audit report", context=None)   # -> path written
```

`simulate_dif_data` and `simulate_poly_dif_data` return a **`SimulatedTest`** with `.responses`, `.group`, `.theta`, `.a_ref`, `.b_ref`, `.a_focal`, `.b_focal`, `.dif_items`, `.nonuniform_items`, `.item_names()`, `.truth_frame()`.

The pipeline simulators return a dict with `responses`, `group`, `theta`, `proxy`, `outcome`, `outcome_observed`, `dif_items`, `train_mask`.

---

## 6. Operation Guide

### 6.1 Basic DIF Analysis

```python
from difair import simulate_dif_data, detect_dif

sim = simulate_dif_data(n_ref=1500, n_focal=1500, n_items=20,
                        n_dif_items=4, dif_magnitude=0.8, seed=7)

res = detect_dif(sim.responses, sim.group, focal_label="focal", purify=True)
print(res.flagged)
print(res.table[["item", "delta_mh", "p_value", "ets_class", "std_p_dif"]].head())
```

> **`focal_label` is required whenever the group labels are not 0/1.** DIF statistics are directional; silently guessing which group is focal would reverse every sign in the output.

### 6.2 Polytomous Items

```python
from difair import simulate_poly_dif_data, detect_dif_poly

sim = simulate_poly_dif_data(n_items=15, n_categories=5,
                             n_dif_items=3, dif_magnitude=0.7, seed=11)
res = detect_dif_poly(sim.responses, sim.group, focal_label="focal",
                      methods=("gmh",), purify=True)
print(res.flagged)
```

### 6.3 Survey Weights

```python
from difair import mantel_haenszel, normalize_weights

w = normalize_weights(raw_weights)          # rescale to sum to the sample size
res = mantel_haenszel(responses, group, focal_label="Y", weights=w)
```

Weighted cells give design-consistent point estimates. Variance formulas still assume simple random sampling, so p-values are anti-conservative under a clustered design and should be read as indicative. Normalizing first keeps the chi-square on its nominal scale.

### 6.4 Design-Based Inference

```python
from difair import survey_dif, jackknife_weights

rw = jackknife_weights(weights, strata, psu=school_id)   # or the survey's own
out = survey_dif(responses, group, "focal", weights,
                 replicate_weights=rw, plausible_values=pv,
                 method="fay")                            # PISA; "jackknife" for TIMSS

print(out[["item", "estimate", "se", "ci_low", "ci_high", "fmi"]])
```

The variance constant **must** match how the replicates were built:

```python
# jackknife  -> SE ratio 1.18 vs naive   (correct for TIMSS JK2)
# fay        -> SE ratio 0.16 vs naive   (wrong constant; now raises a warning)
```

`survey_dif` infers the construction from the weights and warns on a mismatch. To check explicitly:

```python
from difair import infer_replicate_design
infer_replicate_design(rw, weights)["method"]   # 'jackknife', 'brr' or 'fay'
```

**Accuracy note.** In a Monte Carlo study with a genuine cluster effect (25 of 80 clusters sampled), jackknife variance ran about 1.49 times the empirical sampling variance; passing `fpc=25/80` brings that to 1.02. Omitting `psu` on a clustered design understated variance by more than an order of magnitude — that is the error worth avoiding. Prefer the assessment's own replicate weights where available.

### 6.5 Pooling Across Analyses

```python
from difair import pool_estimates

pooled = pool_estimates(estimates, standard_errors, method="fixed")
# reports Cochran's Q and I-squared; use method="random" when analyses disagree
```

### 6.6 Ordinal Outcomes

```python
from difair import ordinal_disparity

d = ordinal_disparity(bands, group, focal_label="F")
print(d["probability_superiority"], d["ps_ci_low"], d["ps_ci_high"])
```

The probability of superiority is 0.5 under equal treatment and carries a Hanley–McNeil standard error, so the interval can be read directly. Under a clustered design use `ordinal_disparity_replicate`, which takes the variance from replicate weights instead.

### 6.7 Pipeline Attribution

```python
from difair import simulate_pipeline_data, detect_dif, attribute_stages

d = simulate_pipeline_data(seed=1)
dif = detect_dif(d["responses"], d["group"], focal_label="focal", purify=True)

att = attribute_stages(
    d["responses"], d["group"], "focal", d["outcome"],
    dif_items=dif.flagged, proxy=d["proxy"],
    train_mask=d["train_mask"], train_outcome=d["outcome_observed"],
)
print(att.summary())
```

### 6.8 Full Audit and Report

The complete chain is in `examples/quickstart.py`:

```python
from difair.dif import detect_dif
from difair.fairness import fairness_report
from difair.pipeline import attribute_stages
from difair.report import audit_report
from difair.simulate import simulate_pipeline_data

d = simulate_pipeline_data(n_ref=1500, n_focal=1500, seed=1)

# 1. item level
dif = detect_dif(d["responses"], d["group"], focal_label="focal", purify=True)

# 2. model level
score = d["responses"].to_numpy().sum(axis=1)
pred = (score >= sorted(score)[int(0.6 * len(score))]).astype(int)
fair = fairness_report(d["outcome"], pred, d["group"], "focal")

# 3. stage attribution
att = attribute_stages(
    d["responses"], d["group"], "focal", d["outcome"],
    dif_items=dif.flagged, proxy=d["proxy"],
    train_mask=d["train_mask"], train_outcome=d["outcome_observed"],
)

# 4. report
audit_report("difair_audit.html", dif_result=dif, fairness_table=fair,
             attribution=att,
             context={"Instrument": "Simulated 30-item test",
                      "Focal group": "focal"})
```

### 6.9 Example Scripts

| Script | Purpose |
|---|---|
| `examples/quickstart.py` | End-to-end walkthrough: items → model → attribution → report |
| `examples/validation_study.py` | Reproduces the simulation study reported in the paper (`--quick` for fewer replications) |
| `examples/crossvalidate_difR.py` | Cross-validates every procedure against difR's R sources |
| `examples/crossvalidate_difR.R` | The R side of the cross-validation |
| `examples/realdata_validation.py` | Reruns the audit on Verbal Aggression and OULAD |
| `examples/timss_validation.py` | Reconstructs the TIMSS 2019 JK2 replicate design and compares design-based with naive standard errors |
| `examples/timss_pooling.py` | Pools item estimates across five TIMSS 2019 countries |

---

## 7. DIFair Studio (Browser Tool)

### 7.1 Opening the Tool

Double-click `difair_studio.html`, or open it from the browser's File menu. No installation, no Python, no command line, and no network connection is required. **All computation happens locally in the browser; nothing is uploaded anywhere.** The page works offline and is therefore safe for real student or respondent data.

### 7.2 The Five Tabs

| Tab | Procedures |
|---|---|
| **Dichotomous DIF** | Mantel–Haenszel, standardization, Breslow–Day, logistic regression DIF, with purification |
| **Polytomous DIF** | Generalized Mantel–Haenszel, ordinal logistic regression |
| **Fairness Metrics** | Demographic parity, equalized odds, equal opportunity, predictive parity, calibration gap, ordinal disparity |
| **Survey Design** | Paired-jackknife replicate weights from a stratum/PSU design, design-based standard errors, and a pooling calculator for combining estimates across analyses |
| **Pipeline Attribution** | Shapley attribution of a decision disparity to the item, sampling or model stage |

### 7.3 Workflow

Each tab follows the same three steps:

1. **Prepare your data** — click **Download data template** to get a CSV showing the exact format the tab expects, or **Load sample data** to generate synthetic data in-browser and try the tab immediately.
2. **Configure the analysis** — set the focal group, matching options, purification, significance level and any design parameters.
3. **Results** — the table appears in the page and can be read or copied directly.

The Survey Design tab additionally offers a **Pool estimates across analyses** panel.

### 7.4 Accuracy Relative to the Python Package

Every procedure in the Studio is a from-scratch JavaScript port of the corresponding Python function, checked against its actual output.

| Procedures | Agreement with Python |
|---|---|
| Mantel–Haenszel, standardization, Breslow–Day, generalized Mantel–Haenszel, all fairness metrics, survey/jackknife functions | 5+ decimal places |
| Item-level logistic DIF, ordinal logistic DIF, pipeline attribution | Very close but not bit-identical — these fit their own solver rather than statsmodels/scikit-learn |

The difference matters most for pipeline attribution when sample reweighting is involved. The page states this in its own **Method & limitations** section.

### 7.5 When to Use the Python Package Instead

- You need bit-exact reproducibility with the validated reference implementation
- You want the standalone HTML audit report bundling DIF, fairness and attribution together
- You want to script the analysis across many files at once
- You need exact statsmodels/scikit-learn fits

---

## 8. Validation and Accuracy

### 8.1 Cross-Validation Against difR

`examples/crossvalidate_difR.py` checks every procedure against difR's own R sources and base R's `stats::mantelhaen.test`, over **108 item-level statistics from six datasets**:

| Quantity | Max abs. diff. | Max rel. diff. |
|---|---|---|
| MH chi-square | 5.8e-13 | 2.5e-12 |
| MH common odds ratio | 5.3e-15 | 4.6e-15 |
| MH var(log alpha), RBG | 4.0e-16 | 1.9e-14 |
| Standardized P-DIF | 4.7e-16 | 1.6e-13 |
| Breslow–Day statistic | 5.0e-05 | 1.9e-05 |
| Logistic LRT chi-square | 3.6e-12 | 8.3e-12 |
| Logistic Nagelkerke dR² | 1.5e-15 | 2.2e-12 |

The Breslow–Day residual is a floor imposed by difR, which rounds that statistic to four decimals on output.

```bash
git clone --depth 1 https://github.com/cran/difR /tmp/difR
python examples/crossvalidate_difR.py --difr-dir /tmp/difR/R
```

**On the continuity correction.** Below a raw difference of 0.5 the corrected quantity `(|d| − 0.5)²` rises again as `d` falls, so an item with no group difference receives a non-zero statistic. DIFair floors it at zero by default; pass `clamp_correction=False` to reproduce difR and `mantelhaen.test` exactly. ETS A/B/C classifications agree for 108 of 108 items either way.

### 8.2 Real Data

**Verbal Aggression** (Vansteelandt 2000; 316 respondents, 24 items, distributed with difR). Agreement with difR on real response patterns: 1.9e-14 on the MH chi-square, 5.1e-15 on the common odds ratio, 3.9e-16 on the standardized P-DIF, 1.7e-13 on the logistic LRT. DIFair flags 6 of 24 items at ETS class B or C.

**OULAD** (Kuzilek et al. 2017; 32,593 students; a 1,614-student cohort forms the item matrix):

| Finding | Value |
|---|---|
| Assessments with DIF by disability | 0 of 12 |
| Assessments with DIF by gender | 0 of 12 |
| Demographic parity gap | +0.111 |
| Equalized odds gap | +0.001 |
| Predictive parity gap | 0.000 |
| Attribution: baseline → residual | 0.116 → 0.111 |

Together these are the signature of a base-rate difference rather than a measurement or modelling artefact, and the attribution correctly finds almost nothing to attribute. Reporting the demographic parity gap alone would invite a mitigation that is not warranted here.

### 8.3 TIMSS 2019

`examples/timss_validation.py` reconstructs the JK2 replicate design from the published `JKZONE` and `JKREP`, scores items from the answer keys in the variable labels, and compares design-based with naive standard errors. Analysis is organised by **item block** rather than booklet: each block appears in exactly two of the fourteen booklets, so grouping by block roughly doubles the usable sample.

Across all 14 blocks of three pooled countries — 159 item analyses, about 1,400 students per block, 218 replicate zones — design-based standard errors ran **1.18×** the naive ones (IQR 1.13–1.23), and 14 items showed sex DIF at ETS class B or C, all with design-based intervals excluding zero.

Replicate-design inference is validated on five TIMSS 2019 country files, each with its own sampling design (69–75 zones, 45–55 students per zone), and recognises all five.

`examples/timss_pooling.py` pools across five countries: 137 items estimated in two or more countries, median I² of 0.000 with 15 items above 0.5, and 36 items showing sex DIF consistent enough that the pooled interval excludes zero.

### 8.4 Simulation Recovery

**Dichotomous** (20 items, 4 with planted DIF, 200 replications):

| DIF magnitude | N per group | MH power | MH Type I |
|---|---|---|---|
| 0.50 | 1500 | 0.693 | 0.006 |
| 0.75 | 1500 | 0.965 | 0.028 |
| 1.00 | 1500 | 0.999 | 0.090 |

**Effect of purification** (N = 500):

| DIF magnitude | Type I raw | Type I purified | Power raw | Power purified |
|---|---|---|---|---|
| 0.50 | 0.043 | 0.013 | 0.645 | 0.740 |
| 0.75 | 0.095 | 0.010 | 0.930 | 0.970 |
| 1.00 | 0.171 | 0.009 | 0.990 | 0.995 |

**Polytomous**, generalized Mantel–Haenszel on 5-category items (20 items, 4 with DIF, 1,000 per group, 50 replications):

| DIF magnitude | Power | Type I | Mean \|SMD\| |
|---|---|---|---|
| 0.00 | 0.000 | 0.000 | 0.011 |
| 0.30 | 0.440 | 0.000 | 0.167 |
| 0.60 | 1.000 | 0.005 | 0.330 |
| 0.90 | 1.000 | 0.069 | 0.480 |

**Impact is not mistaken for DIF**: with an impact of 1.0 logits and no planted DIF, the false-flag rate is 0.0032.

### 8.5 Scope and Limitations

- Pipeline attribution accepts dichotomous and ordered categorical responses; continuous responses are rejected.
- Replicate-weight standard errors are conservative under clustering; the assessment's own replicates are preferable where they exist.
- Two groups per analysis; run pairwise for more.
- Attribution reflects the mitigations implemented here and the causal structure they assume. It answers *"which stage does this mitigation set address?"*, not *"what is the causal effect of group membership?"*.
- Purification assumes flagged items are a minority. In simulation it recovers the planted set exactly with no false positives while up to roughly a quarter of items carry DIF, but breaks down at half.
- **Statistical DIF is not by itself evidence of unfairness.** Flagged items require content review; the package produces evidence for expert judgement, not verdicts.

---

## 9. Troubleshooting

### 9.1 Every DIF Sign Is Reversed

**Cause:** `focal_label` was omitted and the group labels are not 0/1, so the focal group was inferred by string ordering.

**Fix:** always pass `focal_label` explicitly:

```python
detect_dif(responses, group, focal_label="focal")
```

### 9.2 Chi-Square Statistics Are Implausibly Large

**Cause:** survey weights are on the population scale, where one respondent may stand for thousands of students. The statistic is inflated by the scale factor alone.

**Fix:**

```python
from difair import normalize_weights
w = normalize_weights(raw_weights)
```

### 9.3 Standard Errors Look Far Too Small

**Cause (a):** the wrong variance constant. Applying Fay's constant to TIMSS JK2 replicates shrinks standard errors to about a sixth of their proper size.

**Fix:** check the construction before trusting the output:

```python
from difair import infer_replicate_design
infer_replicate_design(rw, weights)["method"]
```

**Cause (b):** `psu` was omitted in `jackknife_weights` on a clustered design, which treats each respondent as their own PSU. This understates variance by more than an order of magnitude.

**Fix:** pass the clustering unit, typically the school:

```python
rw = jackknife_weights(weights, strata, psu=school_id)
```

### 9.4 Standard Errors Look Somewhat Too Large

**Cause:** finite-population conservatism. With 25 of 80 clusters sampled, jackknife variance ran about 1.49× the empirical sampling variance.

**Fix:** pass the sampling fraction:

```python
replicate_variance(est, rep_est, method="jackknife", fpc=25/80)   # ratio -> 1.02
```

### 9.5 Results Do Not Match difR Exactly

**Cause:** the continuity-correction clamp. DIFair floors the corrected statistic at zero; difR does not.

**Fix:**

```python
mantel_haenszel(responses, group, focal_label="F", clamp_correction=False)
```

ETS A/B/C classifications agree either way.

### 9.6 Ordinal Logistic Items Return No Result

**Cause:** the model failed to converge, typically through quasi-complete separation in a sparse category.

**Fix:** collapse low-frequency categories, or use `methods=("gmh",)`, which does not fit a model.

### 9.7 Purification Flags Almost Every Item

**Cause:** purification rebuilds the matching criterion from items believed clean, which requires enough of them to remain. It breaks down when roughly half the items carry DIF.

**Fix:** inspect `res.summary()` before purifying; if most items are flagged, an external matching criterion (`matching=`) is more appropriate.

### 9.8 `attribute_stages` Rejects the Responses

**Cause:** continuous response values. Attribution accepts dichotomous 0/1 codes and ordered category codes only.

**Fix:** discretise the responses, or score them to categories first.

### 9.9 Studio Results Differ Slightly From Python

**Expected** for the logistic-regression-based procedures (item-level logistic DIF, ordinal logistic DIF, pipeline attribution), which fit their own solver rather than statsmodels/scikit-learn. All other procedures agree to 5+ decimal places. Use the Python package when bit-exact reproducibility is required.

### 9.10 `crossvalidate_difR.py` Cannot Find difR

**Fix:** clone the sources and point the script at the `R` directory. Only base R is needed; difR's package tree does not have to be installed.

```bash
git clone --depth 1 https://github.com/cran/difR /tmp/difR
python examples/crossvalidate_difR.py --difr-dir /tmp/difR/R
```

---

## 10. Support and Version Information

### 10.1 Contact

- **Repository:** https://github.com/ZIYINGJERRY/difair
- **Issue tracker:** https://github.com/ZIYINGJERRY/difair/issues
- **Archived release (DOI):** https://doi.org/10.5281/zenodo.21941248

### 10.2 Version

- **Current version:** 0.7.0
- **Released:** 2026-08-15
- **DOI (v0.7.0):** 10.5281/zenodo.21941248
- **Development status:** Alpha
- **License:** MIT

### 10.3 Authors

| Name | Affiliation |
|---|---|
| Ziying Guo | Beijing University of Technology, Beijing, China |
| Yan Li (maintainer, contact) | Beijing Vocational College of Finance and Commerce, Beijing, China |

### 10.4 Citation

If you use DIFair in published work, please cite both the software and the accompanying software paper. Machine-readable metadata is in `CITATION.cff`.

### 10.5 Testing

```bash
pytest tests/ -q --cov=difair
```

Continuous integration runs the full suite on Ubuntu, macOS and Windows across Python 3.9, 3.11 and 3.12, and separately executes `examples/quickstart.py` and verifies the audit report is written.

---

## 11. Appendix

### A. Complete Public API

**Item-level DIF:** `mantel_haenszel`, `logistic_dif`, `standardization`, `breslow_day`, `purify_matching_score`, `detect_dif`, `DIFResult`, `normalize_weights`

**Polytomous items:** `generalized_mantel_haenszel`, `ordinal_logistic_dif`, `purify_poly_matching`, `detect_dif_poly`

**Model-level fairness:** `group_rates`, `demographic_parity`, `equalized_odds`, `equal_opportunity`, `predictive_parity`, `calibration_gap`, `fairness_report`, `ordinal_disparity`, `ordinal_disparity_replicate`, `ordinal_group_summary`

**Pipeline:** `attribute_stages`, `AttributionResult`, `DEFAULT_STAGES`

**Design-based inference:** `survey_dif`, `replicate_variance`, `combine_plausible_values`, `infer_replicate_design`, `jackknife_weights`, `pool_estimates`

**Simulation and reporting:** `simulate_dif_data`, `simulate_pipeline_data`, `simulate_poly_dif_data`, `simulate_poly_pipeline_data`, `SimulatedTest`, `audit_report`

### B. Output Column Glossary

| Column | Meaning |
|---|---|
| `delta_mh` | Mantel–Haenszel statistic on the ETS delta scale. Negative values indicate DIF against the focal group |
| `ets_class` | ETS classification: **A** negligible, **B** intermediate, **C** large |
| `std_p_dif` | Dorans–Kulick standardized proportion difference. `< 0.05` negligible, `0.05–0.10` moderate, `≥ 0.10` large |
| `chi2_uniform` / `p_uniform` | Likelihood-ratio test for uniform DIF (M1 vs M0), 1 df |
| `chi2_nonuniform` / `p_nonuniform` | Likelihood-ratio test for non-uniform DIF (M2 vs M1), 1 df |
| `chi2_total` / `p_total` | Joint test (M2 vs M0), 2 df |
| `delta_r2` | Zumbo–Thomas Nagelkerke effect size |
| `zt_class` | Zumbo–Thomas effect-size classification |
| `smd` | Zwick–Thayer standardized mean difference (polytomous) |
| `favors` | Which group the item favours |
| `n_strata` | Number of matching strata contributing |
| `estimate`, `se`, `ci_low`, `ci_high` | Design-based point estimate and interval |
| `fmi` | Fraction of missing information. A large value means the result is driven by uncertainty about the latent trait rather than by sampling |

### C. Repository Layout

```
difair/
├── difair/                     # package source
│   ├── __init__.py             # public API
│   ├── dif.py                  # dichotomous DIF
│   ├── poly.py                 # polytomous DIF
│   ├── survey.py               # design-based inference
│   ├── fairness.py             # fairness metrics
│   ├── pipeline.py             # Shapley attribution
│   ├── simulate.py             # response simulation
│   └── report.py               # HTML audit report
├── difair_studio.html          # zero-install browser tool
├── examples/                   # runnable walkthroughs and validation scripts
├── tests/                      # pytest suite
├── xval_data/                  # cross-validation datasets
├── validation_results/         # published validation outputs
├── .github/workflows/ci.yml    # CI matrix
├── pyproject.toml
├── CITATION.cff
├── LICENSE                     # MIT
├── README.md
└── README_START_HERE.md
```

### D. Browser Compatibility (DIFair Studio)

| Browser | Version | Support |
|---|---|---|
| Chrome | 80+ | Full |
| Firefox | 75+ | Full |
| Edge | 80+ | Full |
| Safari | 13+ | Full |
| Internet Explorer | — | Not supported |

---

*DIFair 0.7.0 · MIT License · Copyright © 2026 Ziying Guo, Yan Li*
