"""DIFair: unified differential item functioning and algorithmic fairness auditing.

Item-level DIF procedures from educational measurement and model-level fairness
metrics from machine learning, behind one API, plus exact Shapley attribution of
a decision disparity to the pipeline stage that produced it. Dichotomous and
polytomous items are both supported, with survey weights, replicate-weight
variance estimation and plausible values, validated on TIMSS 2019 data.

Quick start
-----------
>>> from difair import simulate_dif_data, detect_dif
>>> sim = simulate_dif_data(n_items=20, n_dif_items=4, dif_magnitude=0.8, seed=7)
>>> res = detect_dif(sim.responses, sim.group, focal_label="focal", purify=True)
>>> len(res.flagged)
4
"""

from difair.dif import (
    DIFResult,
    breslow_day,
    detect_dif,
    logistic_dif,
    mantel_haenszel,
    normalize_weights,
    purify_matching_score,
    standardization,
)
from difair.fairness import (
    calibration_gap,
    demographic_parity,
    equal_opportunity,
    equalized_odds,
    fairness_report,
    group_rates,
    ordinal_disparity,
    ordinal_disparity_replicate,
    ordinal_group_summary,
    predictive_parity,
)
from difair.pipeline import DEFAULT_STAGES, AttributionResult, attribute_stages
from difair.poly import (
    detect_dif_poly,
    generalized_mantel_haenszel,
    ordinal_logistic_dif,
    purify_poly_matching,
)
from difair.report import audit_report
from difair.simulate import (
    SimulatedTest,
    simulate_dif_data,
    simulate_pipeline_data,
    simulate_poly_dif_data,
    simulate_poly_pipeline_data,
)
from difair.survey import (
    combine_plausible_values,
    infer_replicate_design,
    jackknife_weights,
    pool_estimates,
    replicate_variance,
    survey_dif,
)

__version__ = "0.7.0"

__all__ = [
    # item level
    "mantel_haenszel",
    "logistic_dif",
    "standardization",
    "breslow_day",
    "purify_matching_score",
    "detect_dif",
    "DIFResult",
    "normalize_weights",
    # polytomous items
    "generalized_mantel_haenszel",
    "ordinal_logistic_dif",
    "purify_poly_matching",
    "detect_dif_poly",
    # model level
    "group_rates",
    "demographic_parity",
    "equalized_odds",
    "equal_opportunity",
    "predictive_parity",
    "calibration_gap",
    "fairness_report",
    "ordinal_disparity",
    "ordinal_disparity_replicate",
    "ordinal_group_summary",
    # pipeline
    "attribute_stages",
    "AttributionResult",
    "DEFAULT_STAGES",
    # design-based inference
    "survey_dif",
    "replicate_variance",
    "combine_plausible_values",
    "infer_replicate_design",
    "jackknife_weights",
    "pool_estimates",
    # simulation and reporting
    "simulate_dif_data",
    "simulate_pipeline_data",
    "simulate_poly_dif_data",
    "simulate_poly_pipeline_data",
    "SimulatedTest",
    "audit_report",
]
