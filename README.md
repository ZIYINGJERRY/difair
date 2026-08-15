# DIFair

Unified differential item functioning (DIF) and algorithmic fairness auditing
for educational data.

[![CI](https://github.com/ZIYINGJERRY/difair/actions/workflows/ci.yml/badge.svg)](https://github.com/ZIYINGJERRY/difair/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21941248.svg)](https://doi.org/10.5281/zenodo.21941248)

## Why

Educational measurement and machine learning both have a century-old and a
decade-old tradition of asking whether an assessment treats groups fairly, and
the two have almost no software in common. DIF procedures live in R
(`difR`, `mirt`, `difNLR`, `dexter`); algorithmic fairness metrics live in
Python (`fairlearn`, `AIF360`, `aequitas`). A researcher auditing a modern
educational pipeline needs both and currently moves data between languages to
get them.

DIFair puts both in one Python API, and adds the question neither tradition
answers: **given a group gap in the decisions a system makes, which stage of
the pipeline produced it?**

## Two ways to run an audit

| | For | How |
|---|---|---|
| **`difair_studio.html`** | Anyone, no programming | Open the file in a browser. No install, no Python, no network calls. |
| **`difair` (this package)** | Python users, research-grade | `pip install git+https://github.com/ZIYINGJERRY/difair` |

`difair_studio.html` ships alongside this package in the repository rather
than inside the installable Python distribution. Its five tabs cover
dichotomous DIF, polytomous DIF,
fairness metrics, survey design and pipeline attribution, each with a CSV
template and a synthetic sample generator. Every procedure is a from-scratch
JavaScript port checked against the Python implementation; the
Mantel-Haenszel, standardization, Breslow-Day, generalized Mantel-Haenszel,
fairness and survey/jackknife functions match to five or more decimal places,
while the logistic-regression-based procedures use their own solver rather
than statsmodels or scikit-learn and so are close but not bit-identical. See
`README_START_HERE.md` for guidance on which to use.

## Install

DIFair is not on PyPI yet. Install it from this repository:

```bash
pip install git+https://github.com/ZIYINGJERRY/difair
```

Or clone it, which is what you want if you also need `difair_studio.html`,
the examples or the validation data:

```bash
git clone https://github.com/ZIYINGJERRY/difair
cd difair && pip install -e ".[dev]"
```

Python 3.9 or newer. The only requirements are numpy, pandas, scipy,
scikit-learn and statsmodels, all pulled in automatically.

## Quick start

```python
from difair import simulate_dif_data, detect_dif

sim = simulate_dif_data(n_ref=1500, n_focal=1500, n_items=20,
                        n_dif_items=4, dif_magnitude=0.8, seed=7)

res = detect_dif(sim.responses, sim.group, focal_label="focal", purify=True)
print(res.flagged)
print(res.table[["item", "delta_mh", "p_value", "ets_class", "std_p_dif"]].head())
```

`focal_label` is required whenever the group labels are not 0/1: DIF statistics
are directional, and silently guessing which group is focal would reverse every
sign in the output.

## Modules

| Module | Purpose |
|---|---|
| `difair.dif` | Mantel-Haenszel, logistic regression, standardization, Breslow-Day, iterative purification, survey weights |
| `difair.poly` | Generalized Mantel-Haenszel and proportional-odds logistic regression for ordered categorical items |
| `difair.survey` | Replicate-weight variance, plausible values, design-based confidence intervals |
| `difair.fairness` | Demographic parity, equalized odds, equal opportunity, predictive parity, calibration gap |
| `difair.pipeline` | Exact Shapley attribution of a decision disparity to pipeline stages |
| `difair.simulate` | 2PL and generalized partial credit response generation with planted DIF |
| `difair.report` | Standalone HTML audit report |

## Polytomous items

Rating scales and Likert questionnaires use the same API:

```python
from difair import simulate_poly_dif_data, detect_dif_poly

sim = simulate_poly_dif_data(n_items=15, n_categories=5,
                             n_dif_items=3, dif_magnitude=0.7, seed=11)
res = detect_dif_poly(sim.responses, sim.group, focal_label="focal",
                      methods=("gmh",), purify=True)
print(res.flagged)
```

`generalized_mantel_haenszel` reports the Mantel chi-square with the
Zwick-Thayer standardized mean difference; `ordinal_logistic_dif` fits nested
proportional-odds models and separates uniform from non-uniform DIF.

## Survey weights

Large-scale assessments sample with unequal probability. Every stratified
procedure accepts `weights`:

```python
from difair import mantel_haenszel, normalize_weights

w = normalize_weights(raw_weights)          # rescale to sum to the sample size
res = mantel_haenszel(responses, group, focal_label="Y", weights=w)
```

Weighted cells give design-consistent point estimates. Variance formulas still
assume simple random sampling, so p-values are anti-conservative under a
clustered design and should be read as indicative; normalising first keeps the
chi-square on its nominal scale.

## Design-based inference

Large-scale assessments need more than weighted point estimates. `difair.survey`
supplies replicate-weight variance and plausible-value combination:

```python
from difair import survey_dif, jackknife_weights

rw = jackknife_weights(weights, strata, psu=school_id)   # or use the survey's own
out = survey_dif(responses, group, "focal", weights,
                 replicate_weights=rw, plausible_values=pv,
                 method="fay")                            # PISA; "jackknife" for TIMSS

print(out[["item", "estimate", "se", "ci_low", "ci_high", "fmi"]])
```

The variance constant must match how the replicates were built. Applying Fay's
constant to TIMSS JK2 replicates shrinks standard errors to about a sixth of
their proper size, and nothing in the output signals the error — so `survey_dif`
infers the construction from the weights themselves and warns on a mismatch:

```python
# jackknife  -> SE ratio 1.18 vs naive   (correct for TIMSS JK2)
# fay        -> SE ratio 0.16 vs naive   (wrong constant; now raises a warning)
```

```python
from difair import infer_replicate_design
infer_replicate_design(rw, weights)["method"]   # 'jackknife', 'brr' or 'fay'
```

When no signature matches, the return value reports what was observed — the
zeroed, unchanged and doubled fractions and the spread of the non-zero
multipliers — so an unusual design can be told apart from bad input.

The inference is validated on five TIMSS 2019 country files, each with its own
sampling design (69–75 zones, 45–55 students per zone), and recognises all five.

Variance is computed within each plausible value by replication, then combined
across values by Rubin's rules. The reported `fmi` is the fraction of missing
information: a large value means the result is driven by uncertainty about the
latent trait rather than by sampling.

**On accuracy.** In a Monte Carlo study with a genuine cluster effect, 25 of 80
clusters sampled, jackknife variance ran about 1.49 times the empirical
sampling variance. Passing the sampling fraction as `fpc=25/80` brings that to
1.02, essentially removing the conservatism. Omitting `psu` on a clustered
design understated variance by more than an order of magnitude — that is the
error worth avoiding. Prefer the assessment's own replicate weights where
available.

**Validated on TIMSS 2019.** `examples/timss_validation.py` reconstructs the
JK2 replicate design from the published `JKZONE` and `JKREP`, scores items from
the answer keys in the variable labels, and compares design-based with naive
standard errors. Analysis is organised by item block rather than booklet: each
block appears in exactly two of the fourteen booklets, so grouping by block
roughly doubles the usable sample. Across all 14 blocks of three pooled
countries — 159 item analyses, about 1,400 students per block, 218 replicate
zones — design-based standard errors ran 1.18 times the naive ones (IQR
1.13–1.23), and 14 items showed sex DIF at ETS class B or C, all with
design-based intervals excluding zero.

## Pooling across analyses

Independent samples produce separate estimates of the same item. `pool_estimates`
combines them by inverse-variance weighting and reports Cochran's Q and
I-squared, with a random-effects option when the analyses genuinely disagree.

`examples/timss_pooling.py` runs this across five TIMSS 2019 countries: 137
items estimated in two or more countries, median I-squared 0.000 with 15 items
above 0.5, and 36 items showing sex DIF consistent enough that the pooled
interval excludes zero. Items that agree across educational systems pool
cleanly; the ones that do not are exactly what cross-national DIF research is
looking for.

## Ordinal outcomes

Binary fairness metrics collapse an ordered decision to a single cut. For
graded outcomes `ordinal_disparity` reports the mean and standardized
differences, the probability that a focal case outranks a reference case, and
the worst gap any threshold would reveal:

```python
from difair import ordinal_disparity

d = ordinal_disparity(bands, group, focal_label="F")
print(d["probability_superiority"], d["ps_ci_low"], d["ps_ci_high"])
```

The probability of superiority is 0.5 under equal treatment and carries a
Hanley-McNeil standard error, so the interval can be read directly. In
simulation the analytic error tracks the empirical spread to within 10%. Under
a clustered design use `ordinal_disparity_replicate`, which takes the variance
from replicate weights instead.

## Pipeline attribution

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

Each stage is a player in a cooperative game whose characteristic function is
the disparity remaining after that subset of mitigations is applied. With three
or four stages the Shapley values are computed exactly, not sampled.

## Validation

### Cross-validation against difR

`examples/crossvalidate_difR.py` checks every procedure against difR's own R
sources and against base R's `stats::mantelhaen.test`, over 108 item-level
statistics from six datasets:

| Quantity | Max abs. diff. | Max rel. diff. |
|---|---|---|
| MH chi-square | 5.8e-13 | 2.5e-12 |
| MH common odds ratio | 5.3e-15 | 4.6e-15 |
| MH var(log alpha), RBG | 4.0e-16 | 1.9e-14 |
| Standardized P-DIF | 4.7e-16 | 1.6e-13 |
| Breslow-Day statistic | 5.0e-05 | 1.9e-05 |
| Logistic LRT chi-square | 3.6e-12 | 8.3e-12 |
| Logistic Nagelkerke dR2 | 1.5e-15 | 2.2e-12 |

The Breslow-Day residual is a floor imposed by difR, which rounds that
statistic to four decimals on output.

```bash
git clone --depth 1 https://github.com/cran/difR /tmp/difR
python examples/crossvalidate_difR.py --difr-dir /tmp/difR/R
```

Only base R is needed. difR's core computational functions depend on `stats`
alone, so they are sourced directly rather than installing the package and its
deep import tree.

**On the continuity correction.** Below a raw difference of 0.5 the corrected
quantity `(|d| - 0.5)^2` rises again as `d` falls, so an item with no group
difference receives a non-zero statistic. DIFair floors it at zero by default;
pass `clamp_correction=False` to reproduce difR and `mantelhaen.test` exactly.
ETS A/B/C classifications agree for 108 of 108 items either way.

### Real data

`examples/realdata_validation.py` reruns both halves on public data.

**Verbal aggression data** (Vansteelandt 2000; 316 respondents, 24 items,
distributed with difR). Agreement with difR is unchanged on real response
patterns: 1.9e-14 on the MH chi-square, 5.1e-15 on the common odds ratio,
3.9e-16 on the standardized P-DIF, 1.7e-13 on the logistic LRT. DIFair flags
6 of 24 items at ETS class B or C.

**OULAD** (Kuzilek et al. 2017; 32,593 students). Assessments in a 1,614-student
cohort dichotomised at the pass mark form the item matrix, giving a complete
chain from assessment to at-risk decision:

| Finding | Value |
|---|---|
| Assessments with DIF by disability | 0 of 12 |
| Assessments with DIF by gender | 0 of 12 |
| Demographic parity gap | +0.111 |
| Equalized odds gap | +0.001 |
| Predictive parity gap | 0.000 |
| Attribution: baseline -> residual | 0.116 -> 0.111 |

Together these are the signature of a base-rate difference rather than a
measurement or modelling artefact, and the attribution correctly finds almost
nothing to attribute. Reporting the demographic parity gap alone would invite a
mitigation that is not warranted here.

```bash
python examples/realdata_validation.py --difr-dir /tmp/difR --oulad-dir path/to/oulad
```

### Polytomous recovery

Generalized Mantel-Haenszel on 5-category items, 20 items with 4 carrying DIF,
1,000 per group, 50 replications:

| DIF magnitude | Power | Type I | Mean \|SMD\| |
|---|---|---|---|
| 0.00 | 0.000 | 0.000 | 0.011 |
| 0.30 | 0.440 | 0.000 | 0.167 |
| 0.60 | 1.000 | 0.005 | 0.330 |
| 0.90 | 1.000 | 0.069 | 0.480 |

### Simulation study

`examples/validation_study.py` reproduces the study reported in the paper.
Selected results (20 items, 4 with planted DIF, 200 replications):

| DIF magnitude | N per group | MH power | MH Type I |
|---|---|---|---|
| 0.50 | 1500 | 0.693 | 0.006 |
| 0.75 | 1500 | 0.965 | 0.028 |
| 1.00 | 1500 | 0.999 | 0.090 |

Purification controls the matching-score contamination effect (N = 500):

| DIF magnitude | Type I raw | Type I purified | Power raw | Power purified |
|---|---|---|---|---|
| 0.50 | 0.043 | 0.013 | 0.645 | 0.740 |
| 0.75 | 0.095 | 0.010 | 0.930 | 0.970 |
| 1.00 | 0.171 | 0.009 | 0.990 | 0.995 |

Genuine ability differences are not mistaken for DIF: with an impact of 1.0
logits and no planted DIF, the false-flag rate is 0.0032.

```bash
python examples/validation_study.py          # full study
python examples/validation_study.py --quick  # fewer replications
```

## Scope and limitations

* Pipeline attribution accepts dichotomous and ordered categorical responses
  alike; continuous responses are rejected.
* Replicate-weight standard errors are conservative under clustering; the
  assessment's own replicates are preferable where they exist.
* Two groups per analysis; run pairwise for more.
* Attribution reflects the mitigations implemented here and the causal
  structure they assume. It answers "which stage does this mitigation set
  address?", not "what is the causal effect of group membership?".
* Statistical DIF is not by itself evidence of unfairness. Flagged items
  require content review; the package produces evidence for expert judgement,
  not verdicts.

## Testing

```bash
pytest tests/ -q --cov=difair
```

## Citation

If you use DIFair in published work, please cite the accompanying software
paper (see `CITATION.cff`).

Each release is archived on Zenodo. To cite the exact version you ran, use its
version DOI; v0.7.0 is
[10.5281/zenodo.21941248](https://doi.org/10.5281/zenodo.21941248).

## License

MIT
