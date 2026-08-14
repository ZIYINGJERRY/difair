# DIFair — Start Here

This package has two ways to run a DIF (Differential Item Functioning) and
algorithmic fairness audit. Pick the one that fits who's using it.

## Option A — `difair_studio.html`  (for most people)

A single web page covering the full DIFair toolkit — no install, no command
line, no Python. Double-click the file to open it in your browser. It has five tabs:

| Tab | What it does |
|---|---|
| **Dichotomous DIF** | Right/wrong items: Mantel–Haenszel, standardization, Breslow–Day, logistic regression DIF, with purification |
| **Polytomous DIF** | Rating-scale / partial-credit items: generalized Mantel–Haenszel, ordinal logistic regression |
| **Fairness Metrics** | Demographic parity, equalized odds, equal opportunity, predictive parity, calibration gap, ordinal disparity |
| **Survey Design** | Paired-jackknife replicate weights built from your stratum/PSU design, design-based standard errors, plus a small pooling calculator for combining estimates across analyses |
| **Pipeline Attribution** | Shapley attribution of a decision disparity to the item, sampling, or model stage |

Each tab has its own **"Download data template"** button showing the exact
CSV format it expects, and a **"Load sample data"** button that generates
synthetic data in-browser so you can try it immediately.

Everything runs locally in the browser — nothing is uploaded anywhere, so
it's safe to use with real student/respondent data, and it works offline.

**Accuracy.** Every procedure is a from-scratch JavaScript port of the
corresponding function in the Python package below, checked against its
actual output. Mantel–Haenszel, standardization, Breslow–Day, generalized
Mantel–Haenszel, all fairness metrics, and the survey/jackknife functions
match to 5+ decimal places. The logistic-regression-based procedures
(item-level logistic DIF, ordinal logistic DIF, pipeline attribution) fit
their own solver rather than statsmodels/scikit-learn's, so results are very
close but not bit-identical — this is called out in the page's own
"Method & limitations" section, and matters most for pipeline attribution
when sample reweighting is involved.

## Option B — the `difair` Python package  (research-grade)

The full DIFair Python package — every module (`dif`, `poly`, `survey`,
`fairness`, `pipeline`, `simulate`, `report`), validated against `difR` and
real datasets (Verbal Aggression, OULAD, TIMSS 2019). See `README.md` for
full documentation and `examples/quickstart.py` for a runnable walkthrough.

Run this from inside the folder you downloaded or cloned (DIFair is not on
PyPI yet, so there is no `pip install difair`):

```bash
pip install -e ".[dev]"
python examples/quickstart.py
```

Use this when you need bit-exact reproducibility with the validated
reference implementation, or the one or two things the web page doesn't
cover: an HTML audit report bundling everything together, and exact
statsmodels/scikit-learn fits rather than the browser tool's own solvers.

## Which one do I actually need?

If you're not sure, start with `difair_studio.html` — it now covers the
same analyses as the Python package. Reach for the Python package when you need
a fully reproducible research pipeline, or want to script the analysis
across many files at once.
