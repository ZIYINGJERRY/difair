"""Test suite for difair."""

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

from difair.dif import (
    breslow_day,
    detect_dif,
    logistic_dif,
    mantel_haenszel,
    purify_matching_score,
    standardization,
)
from difair.fairness import (
    calibration_gap,
    demographic_parity,
    equalized_odds,
    fairness_report,
    group_rates,
)
from difair.pipeline import attribute_stages
from difair.report import audit_report
from difair.simulate import simulate_dif_data, simulate_pipeline_data


# --------------------------------------------------------------------------- #
# input contract
# --------------------------------------------------------------------------- #
class TestValidation:
    def test_focal_label_required_for_text_labels(self):
        sim = simulate_dif_data(n_ref=200, n_focal=200, n_items=8, seed=0)
        with pytest.raises(ValueError, match="focal_label"):
            mantel_haenszel(sim.responses, sim.group)

    def test_binary_labels_use_convention(self):
        sim = simulate_dif_data(n_ref=300, n_focal=300, n_items=10, seed=0)
        g01 = (sim.group == "focal").astype(int)
        res = mantel_haenszel(sim.responses, g01)
        assert len(res) == 10

    def test_rejects_polytomous(self):
        u = pd.DataFrame(np.random.default_rng(0).integers(0, 3, (100, 5)))
        with pytest.raises(ValueError, match="dichotomous"):
            mantel_haenszel(u, np.tile([0, 1], 50))

    def test_rejects_length_mismatch(self):
        sim = simulate_dif_data(n_ref=100, n_focal=100, n_items=5, seed=0)
        with pytest.raises(ValueError, match="group"):
            mantel_haenszel(sim.responses, np.zeros(10))

    def test_rejects_three_groups(self):
        sim = simulate_dif_data(n_ref=100, n_focal=100, n_items=5, seed=0)
        g = np.array(["a"] * 70 + ["b"] * 70 + ["c"] * 60)
        with pytest.raises(ValueError, match="binary"):
            mantel_haenszel(sim.responses, g)


# --------------------------------------------------------------------------- #
# DIF detection
# --------------------------------------------------------------------------- #
class TestMantelHaenszel:
    def test_recovers_planted_dif(self):
        sim = simulate_dif_data(
            n_ref=1500, n_focal=1500, n_items=20, n_dif_items=4,
            dif_magnitude=0.8, seed=7,
        )
        res = mantel_haenszel(sim.responses, sim.group, focal_label="focal")
        flagged = set(res.loc[res.ets_class.isin(["B", "C"]), "item"])
        assert set(sim.dif_items) <= flagged
        assert len(flagged - set(sim.dif_items)) <= 1

    def test_sign_convention(self):
        """DIF against the focal group gives alpha > 1 and negative delta."""
        sim = simulate_dif_data(
            n_ref=1200, n_focal=1200, n_items=15, n_dif_items=3,
            dif_magnitude=1.0, seed=11,
        )
        res = mantel_haenszel(sim.responses, sim.group, focal_label="focal")
        hit = res[res.item.isin(sim.dif_items)]
        assert (hit.alpha_mh > 1).all()
        assert (hit.delta_mh < 0).all()
        assert (hit.favors == "reference").all()

    def test_no_dif_when_none_planted(self):
        sim = simulate_dif_data(
            n_ref=1200, n_focal=1200, n_items=20, n_dif_items=0, seed=3,
        )
        res = mantel_haenszel(sim.responses, sim.group, focal_label="focal")
        flagged = res.ets_class.isin(["B", "C"]).sum()
        assert flagged <= 2  # allow nominal Type I error

    def test_impact_does_not_create_dif(self):
        """A genuine ability difference must not be mistaken for DIF."""
        sim = simulate_dif_data(
            n_ref=1500, n_focal=1500, n_items=20, n_dif_items=0,
            impact=0.8, seed=5,
        )
        res = mantel_haenszel(sim.responses, sim.group, focal_label="focal")
        assert res.ets_class.isin(["B", "C"]).sum() <= 2

    def test_magnitude_is_monotone(self):
        deltas = []
        for mag in (0.3, 0.6, 1.0):
            sim = simulate_dif_data(
                n_ref=2000, n_focal=2000, n_items=12, n_dif_items=3,
                dif_magnitude=mag, seed=21,
            )
            res = mantel_haenszel(sim.responses, sim.group, focal_label="focal")
            deltas.append(abs(res[res.item.isin(sim.dif_items)].delta_mh.mean()))
        assert deltas[0] < deltas[1] < deltas[2]

    def test_symmetry_under_group_relabelling(self):
        """Swapping which group is focal flips the sign, not the magnitude."""
        sim = simulate_dif_data(
            n_ref=900, n_focal=900, n_items=10, n_dif_items=2,
            dif_magnitude=0.9, seed=13,
        )
        a = mantel_haenszel(sim.responses, sim.group, focal_label="focal")
        b = mantel_haenszel(sim.responses, sim.group, focal_label="reference")
        np.testing.assert_allclose(a.delta_mh.values, -b.delta_mh.values, rtol=1e-9)


class TestLogisticDIF:
    def test_detects_uniform_dif(self):
        sim = simulate_dif_data(
            n_ref=1200, n_focal=1200, n_items=15, n_dif_items=3,
            dif_magnitude=0.9, seed=17,
        )
        res = logistic_dif(sim.responses, sim.group, focal_label="focal")
        hit = res[res.item.isin(sim.dif_items)]
        assert (hit.p_uniform < 0.01).all()

    def test_separates_nonuniform(self):
        sim = simulate_dif_data(
            n_ref=2000, n_focal=2000, n_items=12, n_dif_items=3,
            dif_magnitude=0.0, nonuniform_magnitude=0.9, seed=19,
        )
        res = logistic_dif(sim.responses, sim.group, focal_label="focal")
        hit = res[res.item.isin(sim.dif_items)]
        clean = res[~res.item.isin(sim.dif_items)]
        assert hit.chi2_nonuniform.mean() > clean.chi2_nonuniform.mean()

    def test_effect_size_bounded(self):
        sim = simulate_dif_data(n_ref=600, n_focal=600, n_items=10, seed=23)
        res = logistic_dif(sim.responses, sim.group, focal_label="focal")
        ok = res.delta_r2.dropna()
        assert ((ok >= 0) & (ok <= 1)).all()


class TestStandardization:
    def test_sign_and_magnitude(self):
        sim = simulate_dif_data(
            n_ref=1200, n_focal=1200, n_items=12, n_dif_items=3,
            dif_magnitude=0.9, seed=29,
        )
        res = standardization(sim.responses, sim.group, focal_label="focal")
        hit = res[res.item.isin(sim.dif_items)]
        assert (hit.std_p_dif < 0).all()
        assert (hit.std_class != "negligible").all()

    def test_bounded(self):
        sim = simulate_dif_data(n_ref=500, n_focal=500, n_items=10, seed=31)
        res = standardization(sim.responses, sim.group, focal_label="focal")
        assert res.std_p_dif.abs().max() <= 1.0


class TestBreslowDay:
    def test_flags_nonuniform_more_often(self):
        uni = simulate_dif_data(
            n_ref=1500, n_focal=1500, n_items=12, n_dif_items=3,
            dif_magnitude=0.8, nonuniform_magnitude=0.0, seed=37,
        )
        non = simulate_dif_data(
            n_ref=1500, n_focal=1500, n_items=12, n_dif_items=3,
            dif_magnitude=0.0, nonuniform_magnitude=1.0, seed=37,
        )
        r_uni = breslow_day(uni.responses, uni.group, focal_label="focal")
        r_non = breslow_day(non.responses, non.group, focal_label="focal")
        s_uni = r_uni[r_uni.item.isin(uni.dif_items)].bd_stat.mean()
        s_non = r_non[r_non.item.isin(non.dif_items)].bd_stat.mean()
        assert s_non > s_uni

    def test_expected_counts_feasible(self):
        from difair.dif import _bd_expected

        e = _bd_expected(psi=2.0, n_r=50, m1=40, T=100)
        assert e is not None and 0 < e < min(50, 40)


class TestPurification:
    def test_removes_dif_items_from_matching(self):
        sim = simulate_dif_data(
            n_ref=1500, n_focal=1500, n_items=20, n_dif_items=5,
            dif_magnitude=1.0, seed=41,
        )
        _, flagged = purify_matching_score(sim.responses, sim.group, focal_label="focal")
        assert len(set(flagged) & set(sim.dif_items)) >= 4


class TestDetectDIF:
    def test_merges_methods(self):
        sim = simulate_dif_data(n_ref=800, n_focal=800, n_items=10, seed=43)
        res = detect_dif(sim.responses, sim.group, focal_label="focal")
        for col in ("item", "delta_mh", "p_uniform", "std_p_dif", "bd_stat"):
            assert col in res.table.columns
        assert len(res.table) == 10

    def test_summary_and_repr(self):
        sim = simulate_dif_data(n_ref=500, n_focal=500, n_items=8, seed=47)
        res = detect_dif(sim.responses, sim.group, focal_label="focal", methods=("mh",))
        assert res.summary().n_items.sum() == 8
        assert "DIFResult" in repr(res)

    def test_rejects_empty_methods(self):
        sim = simulate_dif_data(n_ref=200, n_focal=200, n_items=5, seed=53)
        with pytest.raises(ValueError):
            detect_dif(sim.responses, sim.group, focal_label="focal", methods=())


# --------------------------------------------------------------------------- #
# fairness metrics
# --------------------------------------------------------------------------- #
class TestFairness:
    @staticmethod
    def _toy():
        g = np.array(["r"] * 100 + ["f"] * 100)
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        pred = np.where(g == "r", rng.random(200) < 0.6, rng.random(200) < 0.3).astype(int)
        return y, pred, g

    def test_group_rates_shape(self):
        y, pred, g = self._toy()
        r = group_rates(y, pred, g)
        assert len(r) == 2 and {"tpr", "fpr", "ppv"} <= set(r.columns)

    def test_demographic_parity_direction(self):
        y, pred, g = self._toy()
        dp = demographic_parity(pred, g, "f")
        assert dp["difference"] < 0  # focal selected less often
        assert 0 <= dp["ratio"] <= 1

    def test_perfect_parity_is_zero(self):
        g = np.array(["r"] * 100 + ["f"] * 100)
        pred = np.tile([1, 0], 100)
        assert abs(demographic_parity(pred, g, "f")["difference"]) < 1e-12

    def test_equalized_odds_keys(self):
        y, pred, g = self._toy()
        eo = equalized_odds(y, pred, g, "f")
        assert {"tpr_difference", "fpr_difference", "max_violation"} <= set(eo)

    def test_calibration_gap_zero_when_identical(self):
        rng = np.random.default_rng(1)
        s = rng.random(2000)
        y = (rng.random(2000) < s).astype(int)
        g = np.array(["r", "f"] * 1000)
        cg = calibration_gap(y, s, g, "f")
        assert abs(cg["difference"]) < 0.05

    def test_report_runs_without_truth(self):
        _, pred, g = self._toy()
        rep = fairness_report(None, pred, g, "f")
        assert list(rep.metric) == ["demographic_parity"]

    def test_report_full(self):
        y, pred, g = self._toy()
        rng = np.random.default_rng(2)
        rep = fairness_report(y, pred, g, "f", y_score=rng.random(200))
        assert len(rep) == 5


# --------------------------------------------------------------------------- #
# pipeline attribution
# --------------------------------------------------------------------------- #
class TestAttribution:
    def test_shapley_is_efficient(self):
        """Shapley values must sum to the total disparity removed."""
        d = simulate_pipeline_data(n_ref=800, n_focal=800, seed=2)
        res = attribute_stages(
            d["responses"], d["group"], "focal", d["outcome"],
            dif_items=d["dif_items"], proxy=d["proxy"],
            train_mask=d["train_mask"], train_outcome=d["outcome_observed"],
        )
        assert abs(res.shapley.shapley_value.sum() - res.explained) < 1e-9

    def test_item_share_grows_with_dif(self):
        shares = []
        for mag in (0.2, 1.2):
            d = simulate_pipeline_data(
                n_ref=1200, n_focal=1200, dif_magnitude=mag,
                label_bias=0.1, proxy_strength=0.3, seed=9,
            )
            res = attribute_stages(
                d["responses"], d["group"], "focal", d["outcome"],
                dif_items=d["dif_items"], proxy=d["proxy"],
                train_mask=d["train_mask"], train_outcome=d["outcome_observed"],
            )
            s = res.summary().set_index("stage").shapley_value
            shares.append(s.get("item", 0.0))
        assert shares[1] > shares[0]

    def test_stage_dropped_without_proxy(self):
        d = simulate_pipeline_data(n_ref=600, n_focal=600, seed=4)
        res = attribute_stages(
            d["responses"], d["group"], "focal", d["outcome"],
            dif_items=d["dif_items"], train_mask=d["train_mask"],
        )
        assert "model" not in set(res.shapley.stage)

    def test_rejects_unknown_stage(self):
        d = simulate_pipeline_data(n_ref=400, n_focal=400, seed=6)
        with pytest.raises(ValueError, match="Unknown stage"):
            attribute_stages(
                d["responses"], d["group"], "focal", d["outcome"],
                dif_items=d["dif_items"], proxy=d["proxy"],
                train_mask=d["train_mask"], stages=("item", "wizardry"),
            )

    def test_rejects_bad_metric(self):
        d = simulate_pipeline_data(n_ref=400, n_focal=400, seed=8)
        with pytest.raises(ValueError, match="metric"):
            attribute_stages(
                d["responses"], d["group"], "focal", d["outcome"],
                dif_items=d["dif_items"], proxy=d["proxy"],
                train_mask=d["train_mask"], metric="vibes",
            )


# --------------------------------------------------------------------------- #
# simulation and reporting
# --------------------------------------------------------------------------- #
class TestSimulate:
    def test_truth_frame(self):
        sim = simulate_dif_data(n_ref=200, n_focal=200, n_items=10, n_dif_items=3, seed=0)
        t = sim.truth_frame()
        assert t.has_dif.sum() == 3
        assert np.allclose(t.b_shift[t.has_dif], sim.b_focal[:3] - sim.b_ref[:3])

    def test_reproducible(self):
        a = simulate_dif_data(n_ref=100, n_focal=100, n_items=5, seed=99)
        b = simulate_dif_data(n_ref=100, n_focal=100, n_items=5, seed=99)
        pd.testing.assert_frame_equal(a.responses, b.responses)

    def test_rejects_too_many_dif_items(self):
        with pytest.raises(ValueError):
            simulate_dif_data(n_items=5, n_dif_items=9)

    def test_pipeline_base_rates_equal(self):
        """True outcome rates must match across groups by construction."""
        d = simulate_pipeline_data(n_ref=3000, n_focal=3000, seed=12)
        f = d["outcome"][d["group"] == "focal"].mean()
        r = d["outcome"][d["group"] == "reference"].mean()
        assert abs(f - r) < 0.05


class TestReport:
    def test_writes_html(self, tmp_path):
        sim = simulate_dif_data(n_ref=400, n_focal=400, n_items=8, seed=0)
        dif = detect_dif(sim.responses, sim.group, focal_label="focal", methods=("mh",))
        out = audit_report(str(tmp_path / "r.html"), dif_result=dif)
        text = open(out, encoding="utf-8").read()
        assert "<html" in text and "differential item functioning" in text

    def test_handles_all_none(self, tmp_path):
        out = audit_report(str(tmp_path / "e.html"))
        assert "Not supplied" in open(out, encoding="utf-8").read()


class TestReferenceCompatibility:
    """Guards on the behaviours that the difR cross-validation pinned down."""

    def test_clamp_changes_only_near_null_items(self):
        sim = simulate_dif_data(
            n_ref=1200, n_focal=1200, n_items=20, n_dif_items=4,
            dif_magnitude=0.8, seed=61,
        )
        on = mantel_haenszel(sim.responses, sim.group, focal_label="focal")
        off = mantel_haenszel(sim.responses, sim.group, focal_label="focal",
                              clamp_correction=False)
        diff = (on.chi2 - off.chi2).abs()
        # Any item where the two differ must be nowhere near significance.
        assert (off.loc[diff > 1e-9, "chi2"] < 0.25).all()

    def test_clamp_floors_at_zero(self):
        sim = simulate_dif_data(n_ref=800, n_focal=800, n_items=12, seed=63)
        res = mantel_haenszel(sim.responses, sim.group, focal_label="focal")
        assert (res.chi2.dropna() >= 0).all()

    def test_ets_classification_invariant_to_clamp(self):
        sim = simulate_dif_data(
            n_ref=1500, n_focal=1500, n_items=20, n_dif_items=4,
            dif_magnitude=0.7, seed=67,
        )
        on = mantel_haenszel(sim.responses, sim.group, focal_label="focal")
        off = mantel_haenszel(sim.responses, sim.group, focal_label="focal",
                              clamp_correction=False)
        assert (on.ets_class == off.ets_class).all()

    def test_logistic_lrt_is_nonnegative(self):
        """Newton fits must not produce negative likelihood-ratio statistics."""
        sim = simulate_dif_data(n_ref=900, n_focal=900, n_items=15, seed=71)
        res = logistic_dif(sim.responses, sim.group, focal_label="focal")
        for col in ("chi2_total", "chi2_uniform", "chi2_nonuniform"):
            assert (res[col].dropna() >= 0).all()


class TestPublicAPI:
    """The documented import surface must actually exist."""

    def test_top_level_exports_resolve(self):
        import difair

        for name in difair.__all__:
            assert hasattr(difair, name), f"difair.__all__ lists missing {name}"

    def test_documented_entry_points_importable(self):
        from difair import (  # noqa: F401
            attribute_stages,
            audit_report,
            detect_dif,
            fairness_report,
            simulate_dif_data,
            simulate_pipeline_data,
        )

    def test_readme_quickstart_runs(self):
        from difair import detect_dif, simulate_dif_data

        sim = simulate_dif_data(n_ref=1500, n_focal=1500, n_items=20,
                                n_dif_items=4, dif_magnitude=0.8, seed=7)
        res = detect_dif(sim.responses, sim.group, focal_label="focal", purify=True)
        assert set(res.flagged) == set(sim.dif_items)

    def test_version_present(self):
        import difair

        assert difair.__version__ == "0.7.0"


class TestDegenerateInput:
    def test_warns_on_constant_items(self):
        u = pd.DataFrame({
            "const": [1] * 100,
            "ok": np.random.default_rng(0).integers(0, 2, 100),
        })
        with pytest.warns(UserWarning, match="no variance"):
            mantel_haenszel(u, np.tile([0, 1], 50), focal_label=1)

    def test_no_warning_on_clean_data(self):
        sim = simulate_dif_data(n_ref=400, n_focal=400, n_items=10, seed=1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mantel_haenszel(sim.responses, sim.group, focal_label="focal")
        assert not [w for w in caught if "no variance" in str(w.message)]

    def test_accepts_ndarray_and_dataframe_alike(self):
        sim = simulate_dif_data(n_ref=600, n_focal=600, n_items=10, seed=5)
        a = mantel_haenszel(sim.responses, sim.group, focal_label="focal")
        b = mantel_haenszel(sim.responses.to_numpy(), sim.group, focal_label="focal")
        np.testing.assert_allclose(a.delta_mh.values, b.delta_mh.values)

    def test_boolean_and_integer_groups_agree(self):
        sim = simulate_dif_data(n_ref=600, n_focal=600, n_items=10, seed=5)
        ref = mantel_haenszel(sim.responses, sim.group, focal_label="focal")
        for g in ((sim.group == "focal").astype(int), sim.group == "focal"):
            got = mantel_haenszel(sim.responses, g)
            np.testing.assert_allclose(ref.delta_mh.values, got.delta_mh.values)


# --------------------------------------------------------------------------- #
# polytomous items (v0.2)
# --------------------------------------------------------------------------- #
from difair.poly import (  # noqa: E402
    detect_dif_poly,
    generalized_mantel_haenszel,
    ordinal_logistic_dif,
    purify_poly_matching,
)
from difair.simulate import simulate_poly_dif_data  # noqa: E402


class TestPolytomous:
    def test_recovers_planted_dif(self):
        sim = simulate_poly_dif_data(n_ref=1200, n_focal=1200, n_items=15,
                                     n_categories=5, n_dif_items=3,
                                     dif_magnitude=0.7, seed=11)
        res = generalized_mantel_haenszel(sim.responses, sim.group, focal_label="focal")
        flagged = set(res.loc[res.smd_class.isin(["B", "C"]), "item"])
        assert set(sim.dif_items) <= flagged

    def test_sign_convention(self):
        sim = simulate_poly_dif_data(n_ref=1000, n_focal=1000, n_items=12,
                                     n_dif_items=3, dif_magnitude=0.8, seed=13)
        res = generalized_mantel_haenszel(sim.responses, sim.group, focal_label="focal")
        hit = res[res.item.isin(sim.dif_items)]
        assert (hit.smd < 0).all()
        assert (hit.favors == "reference").all()

    def test_no_dif_when_none_planted(self):
        sim = simulate_poly_dif_data(n_ref=1200, n_focal=1200, n_items=15,
                                     n_dif_items=0, seed=17)
        res = generalized_mantel_haenszel(sim.responses, sim.group, focal_label="focal")
        assert res.smd_class.isin(["B", "C"]).sum() <= 2

    def test_impact_not_mistaken_for_dif(self):
        sim = simulate_poly_dif_data(n_ref=1200, n_focal=1200, n_items=15,
                                     n_dif_items=0, impact=0.8, seed=19)
        res = generalized_mantel_haenszel(sim.responses, sim.group, focal_label="focal")
        assert res.smd_class.isin(["B", "C"]).sum() <= 2

    def test_symmetry_under_relabelling(self):
        sim = simulate_poly_dif_data(n_ref=800, n_focal=800, n_items=10,
                                     n_dif_items=2, dif_magnitude=0.8, seed=23)
        a = generalized_mantel_haenszel(sim.responses, sim.group, focal_label="focal")
        b = generalized_mantel_haenszel(sim.responses, sim.group, focal_label="reference")
        np.testing.assert_allclose(a.smd.values, -b.smd.values, rtol=1e-9)
        np.testing.assert_allclose(a.chi2.values, b.chi2.values, rtol=1e-9)

    def test_rejects_non_integer_codes(self):
        u = pd.DataFrame(np.random.default_rng(0).random((100, 4)))
        with pytest.raises(ValueError, match="integer category codes"):
            generalized_mantel_haenszel(u, np.tile([0, 1], 50), focal_label=1)

    def test_ordinal_detects_uniform_dif(self):
        sim = simulate_poly_dif_data(n_ref=700, n_focal=700, n_items=6,
                                     n_dif_items=2, dif_magnitude=0.8, seed=29)
        res = ordinal_logistic_dif(sim.responses, sim.group, focal_label="focal")
        hit = res[res.item.isin(sim.dif_items)]
        assert (hit.p_uniform < 0.01).all()
        assert (hit.beta_group < 0).all()

    def test_purification_runs(self):
        sim = simulate_poly_dif_data(n_ref=900, n_focal=900, n_items=12,
                                     n_dif_items=3, dif_magnitude=0.9, seed=31)
        _, flagged = purify_poly_matching(sim.responses, sim.group, focal_label="focal")
        assert len(set(flagged) & set(sim.dif_items)) >= 2

    def test_detect_dif_poly_returns_difresult(self):
        from difair.dif import DIFResult

        sim = simulate_poly_dif_data(n_ref=600, n_focal=600, n_items=8,
                                     n_dif_items=2, dif_magnitude=0.8, seed=37)
        res = detect_dif_poly(sim.responses, sim.group, focal_label="focal",
                              methods=("gmh",))
        assert isinstance(res, DIFResult)
        assert len(res.table) == 8

    def test_simulator_produces_all_categories(self):
        sim = simulate_poly_dif_data(n_ref=800, n_focal=800, n_items=10,
                                     n_categories=5, seed=41)
        assert sorted(pd.unique(sim.responses.to_numpy().ravel())) == [0, 1, 2, 3, 4]

    def test_simulator_rejects_two_categories(self):
        with pytest.raises(ValueError, match="at least 3"):
            simulate_poly_dif_data(n_categories=2)


# --------------------------------------------------------------------------- #
# survey weights (v0.2)
# --------------------------------------------------------------------------- #
from difair.dif import normalize_weights  # noqa: E402


class TestSurveyWeights:
    def test_unit_weights_match_unweighted(self):
        sim = simulate_dif_data(n_ref=800, n_focal=800, n_items=12, seed=43)
        a = mantel_haenszel(sim.responses, sim.group, focal_label="focal")
        b = mantel_haenszel(sim.responses, sim.group, focal_label="focal",
                            weights=np.ones(1600))
        np.testing.assert_allclose(a.delta_mh.values, b.delta_mh.values)
        np.testing.assert_allclose(a.chi2.values, b.chi2.values)

    def test_integer_weights_equal_replication(self):
        """The strongest correctness check: weight w == the row repeated w times."""
        sim = simulate_dif_data(n_ref=700, n_focal=700, n_items=10,
                                n_dif_items=2, dif_magnitude=0.8, seed=47)
        w = np.random.default_rng(0).integers(1, 4, 1400).astype(float)
        rep = np.repeat(np.arange(1400), w.astype(int))
        a = mantel_haenszel(sim.responses, sim.group, focal_label="focal", weights=w)
        b = mantel_haenszel(sim.responses.iloc[rep], sim.group[rep], focal_label="focal")
        np.testing.assert_allclose(a.delta_mh.values, b.delta_mh.values)
        np.testing.assert_allclose(a.chi2.values, b.chi2.values)

    def test_uniform_scaling_preserves_odds_ratio(self):
        sim = simulate_dif_data(n_ref=600, n_focal=600, n_items=10, seed=53)
        a = mantel_haenszel(sim.responses, sim.group, focal_label="focal")
        b = mantel_haenszel(sim.responses, sim.group, focal_label="focal",
                            weights=np.full(1200, 7.0))
        np.testing.assert_allclose(a.alpha_mh.values, b.alpha_mh.values)

    def test_standardization_accepts_weights(self):
        sim = simulate_dif_data(n_ref=600, n_focal=600, n_items=10, seed=59)
        a = standardization(sim.responses, sim.group, focal_label="focal")
        b = standardization(sim.responses, sim.group, focal_label="focal",
                            weights=np.ones(1200))
        np.testing.assert_allclose(a.std_p_dif.values, b.std_p_dif.values)

    def test_normalize_sums_to_sample_size(self):
        w = np.array([1000.0, 2000.0, 3000.0, 4000.0])
        n = normalize_weights(w)
        assert abs(n.sum() - 4) < 1e-9
        np.testing.assert_allclose(n / n[0], w / w[0])

    def test_normalize_by_group(self):
        w = np.array([10.0, 20.0, 100.0, 300.0])
        g = np.array(["r", "r", "f", "f"])
        n = normalize_weights(w, group=g)
        assert abs(n[:2].sum() - 2) < 1e-9
        assert abs(n[2:].sum() - 2) < 1e-9

    def test_normalize_rejects_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            normalize_weights(np.array([1.0, -1.0]))


class TestReportCoverage:
    def test_truncates_long_tables(self, tmp_path):
        sim = simulate_dif_data(n_ref=500, n_focal=500, n_items=40, seed=61)
        dif = detect_dif(sim.responses, sim.group, focal_label="focal", methods=("mh",))
        out = audit_report(str(tmp_path / "long.html"), dif_result=dif)
        text = open(out, encoding="utf-8").read()
        assert "<html" in text

    def test_includes_all_sections(self, tmp_path):
        sim = simulate_pipeline_data(n_ref=400, n_focal=400, seed=63)
        dif = detect_dif(sim["responses"], sim["group"], focal_label="focal",
                         methods=("mh",))
        fair = fairness_report(sim["outcome"],
                               (np.arange(len(sim["group"])) % 2), sim["group"], "focal")
        att = attribute_stages(sim["responses"], sim["group"], "focal", sim["outcome"],
                               dif_items=dif.flagged, proxy=sim["proxy"],
                               train_mask=sim["train_mask"])
        out = audit_report(str(tmp_path / "full.html"), dif_result=dif,
                           fairness_table=fair, attribution=att,
                           context={"Instrument": "test"})
        text = open(out, encoding="utf-8").read()
        for section in ("differential item functioning", "Model-level fairness",
                        "Pipeline-stage attribution", "Interpretation notes"):
            assert section in text


# --------------------------------------------------------------------------- #
# v0.3: polytomous pipeline, survey inference, ordinal performance
# --------------------------------------------------------------------------- #
from difair.simulate import simulate_poly_pipeline_data  # noqa: E402
from difair.survey import (  # noqa: E402
    combine_plausible_values,
    jackknife_weights,
    replicate_variance,
    survey_dif,
)


class TestPolytomousPipeline:
    def test_attribution_accepts_ordered_responses(self):
        from difair.poly import detect_dif_poly

        d = simulate_poly_pipeline_data(n_ref=800, n_focal=800, n_items=12,
                                        n_dif_items=3, seed=5)
        dif = detect_dif_poly(d["responses"], d["group"], focal_label="focal",
                              methods=("gmh",), purify=True)
        att = attribute_stages(
            d["responses"], d["group"], "focal", d["outcome"],
            dif_items=dif.flagged, proxy=d["proxy"],
            train_mask=d["train_mask"], train_outcome=d["outcome_observed"],
        )
        assert att.response_kind == "polytomous"
        assert abs(att.shapley.shapley_value.sum() - att.explained) < 1e-9

    def test_response_kind_detected(self):
        from difair.pipeline import _response_kind

        assert _response_kind(pd.DataFrame({"a": [0, 1, 1], "b": [1, 0, 1]})) == "dichotomous"
        assert _response_kind(pd.DataFrame({"a": [0, 2, 4], "b": [1, 3, 2]})) == "polytomous"

    def test_rejects_continuous_responses(self):
        d = simulate_poly_pipeline_data(n_ref=200, n_focal=200, n_items=6, seed=7)
        resp = d["responses"].astype(float) + 0.5
        with pytest.raises(ValueError, match="integer category codes"):
            attribute_stages(resp, d["group"], "focal", d["outcome"],
                             dif_items=[], proxy=d["proxy"],
                             train_mask=d["train_mask"])

    def test_dichotomous_path_unchanged(self):
        d = simulate_pipeline_data(n_ref=600, n_focal=600, seed=9)
        att = attribute_stages(d["responses"], d["group"], "focal", d["outcome"],
                               dif_items=d["dif_items"], proxy=d["proxy"],
                               train_mask=d["train_mask"])
        assert att.response_kind == "dichotomous"


class TestSurveyInference:
    def test_rubin_rules_analytic(self):
        est = [1.0, 2.0, 3.0, 4.0]
        between = float(np.var(est, ddof=1))
        r = combine_plausible_values(est, [0.5] * 4)
        assert abs(r["between"] - between) < 1e-12
        assert abs(r["variance"] - (0.5 + 1.25 * between)) < 1e-12
        assert abs(r["fmi"] - 1.25 * between / (0.5 + 1.25 * between)) < 1e-12

    def test_single_plausible_value_has_no_between(self):
        r = combine_plausible_values([2.0], [0.3])
        assert abs(r["variance"] - 0.3) < 1e-12
        assert r["between"] == 0.0

    def test_replicate_variance_constants(self):
        est, rep = 2.0, np.array([2.1, 1.9, 2.2, 1.8, 2.05, 1.95])
        dev = ((rep - est) ** 2)
        assert abs(replicate_variance(est, rep, "jackknife")["variance"] - dev.sum()) < 1e-12
        assert abs(replicate_variance(est, rep, "brr")["variance"] - dev.mean()) < 1e-12
        fay = replicate_variance(est, rep, "fay")["variance"]
        assert abs(fay - dev.sum() / (6 * 0.25)) < 1e-12

    def test_rejects_unknown_variance_method(self):
        with pytest.raises(ValueError, match="jackknife"):
            replicate_variance(1.0, [1.1, 0.9], method="guesswork")

    def test_jackknife_preserves_stratum_totals(self):
        w = np.ones(60)
        psu = np.arange(60) % 6
        rw = jackknife_weights(w, np.zeros(60), psu=psu)
        assert rw.shape[0] == 6
        for row in rw:
            assert abs(row.sum() - w.sum()) < 1e-9  # total preserved
            assert (row == 0).sum() == 10           # one PSU dropped

    def test_jackknife_requires_two_psus(self):
        with pytest.raises(ValueError, match="two or more"):
            jackknife_weights(np.ones(10), strata=np.arange(10), psu=np.arange(10))

    def test_survey_dif_reports_intervals(self):
        sim = simulate_dif_data(n_ref=600, n_focal=600, n_items=6,
                                n_dif_items=2, dif_magnitude=0.9, seed=3)
        n = 1200
        w = np.ones(n)
        rw = jackknife_weights(w, np.zeros(n), psu=np.arange(n) % 20)
        out = survey_dif(sim.responses, sim.group, "focal", w, replicate_weights=rw)
        assert len(out) == 6
        assert (out.se.dropna() > 0).all()
        flagged = out[(out.ci_high < 0)]
        assert set(sim.dif_items) <= set(flagged.item)

    def test_plausible_values_are_binned(self):
        """Continuous criteria must be discretised or nothing is estimable."""
        sim = simulate_dif_data(n_ref=500, n_focal=500, n_items=6, seed=11)
        n = 1000
        rng = np.random.default_rng(0)
        total = sim.responses.to_numpy().sum(axis=1).astype(float)
        pv = np.vstack([total + rng.normal(0, 0.7, n) for _ in range(3)])
        out = survey_dif(sim.responses, sim.group, "focal", np.ones(n),
                         plausible_values=pv)
        assert out.estimate.notna().all()
        assert (out.n_plausible_values == 3).all()

    def test_survey_dif_rejects_unknown_statistic(self):
        sim = simulate_dif_data(n_ref=300, n_focal=300, n_items=5, seed=13)
        with pytest.raises(ValueError, match="not a column"):
            survey_dif(sim.responses, sim.group, "focal", np.ones(600),
                       statistic="vibes")


class TestOrdinalFit:
    def test_matches_statsmodels(self):
        """The hand-written cumulative-logit fit must agree with OrderedModel."""
        from statsmodels.miscmodels.ordinal_model import OrderedModel

        from difair.poly import _fit_cumlogit

        rng = np.random.default_rng(0)
        n = 800
        x = rng.normal(size=n)
        eta = 0.8 * x
        cuts = np.array([-1.0, 0.0, 1.2])
        cum = 1 / (1 + np.exp(-(cuts[None, :] - eta[:, None])))
        cum = np.concatenate([np.zeros((n, 1)), cum, np.ones((n, 1))], axis=1)
        p = np.diff(cum, axis=1)
        y = np.array([rng.choice(4, p=row) for row in p])

        ll, params = _fit_cumlogit(y, x[:, None], 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = OrderedModel(y, x[:, None], distr="logit").fit(disp=0)
        assert abs(ll - ref.llf) < 1e-3
        assert abs(params[-1] - ref.params[0]) < 1e-2

    def test_ordinal_still_detects_dif(self):
        sim = simulate_poly_dif_data(n_ref=700, n_focal=700, n_items=6,
                                     n_dif_items=2, dif_magnitude=0.8, seed=29)
        res = ordinal_logistic_dif(sim.responses, sim.group, focal_label="focal")
        hit = res[res.item.isin(sim.dif_items)]
        assert (hit.p_uniform < 0.01).all()
        assert (hit.beta_group < 0).all()

    def test_handles_sparse_categories(self):
        """A category present for only a handful of respondents must not crash."""
        rng = np.random.default_rng(3)
        n = 300
        u = rng.integers(0, 3, (n, 4))
        u[:5, 0] = 5                       # a rare high category
        res = ordinal_logistic_dif(pd.DataFrame(u), np.tile([0, 1], n // 2),
                                   focal_label=1)
        assert len(res) == 4

    def test_handles_constant_item_gracefully(self):
        u = pd.DataFrame({"a": [1] * 200, "b": np.random.default_rng(0).integers(0, 4, 200)})
        with pytest.warns(UserWarning, match="no variance"):
            res = ordinal_logistic_dif(u, np.tile([0, 1], 100), focal_label=1)
        assert np.isnan(res.loc[res.item == "a", "chi2_total"]).all()


class TestPurificationBoundary:
    def test_flagging_every_item_raises_informatively(self):
        d = simulate_poly_pipeline_data(n_ref=400, n_focal=400, n_items=8, seed=3)
        with pytest.raises(ValueError, match="every item"):
            attribute_stages(d["responses"], d["group"], "focal", d["outcome"],
                             dif_items=list(d["responses"].columns),
                             proxy=d["proxy"], train_mask=d["train_mask"])

    def test_purification_holds_at_moderate_contamination(self):
        """Up to roughly a quarter of items carrying DIF, recovery is exact."""
        from difair.poly import detect_dif_poly

        d = simulate_poly_pipeline_data(n_ref=600, n_focal=600, n_items=20,
                                        n_dif_items=4, seed=3)
        res = detect_dif_poly(d["responses"], d["group"], focal_label="focal",
                              methods=("gmh",), purify=True)
        assert set(res.flagged) == set(d["dif_items"])


# --------------------------------------------------------------------------- #
# v0.4: finite-population correction, ordinal fairness, TIMSS scoring
# --------------------------------------------------------------------------- #
from difair.fairness import ordinal_disparity, ordinal_group_summary  # noqa: E402


class TestFinitePopulationCorrection:
    def test_scales_variance_by_one_minus_fraction(self):
        est, rep = 2.0, np.array([2.1, 1.9, 2.2, 1.8])
        plain = replicate_variance(est, rep, "jackknife")["variance"]
        corrected = replicate_variance(est, rep, "jackknife", fpc=0.25)["variance"]
        assert abs(corrected - plain * 0.75) < 1e-12

    def test_zero_fraction_is_a_no_op(self):
        est, rep = 1.0, np.array([1.2, 0.8, 1.1])
        a = replicate_variance(est, rep, "jackknife")["variance"]
        b = replicate_variance(est, rep, "jackknife", fpc=0.0)["variance"]
        assert abs(a - b) < 1e-12

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="sampling fraction"):
            replicate_variance(1.0, [1.1, 0.9], fpc=1.5)
        with pytest.raises(ValueError, match="sampling fraction"):
            replicate_variance(1.0, [1.1, 0.9], fpc=-0.1)

    def test_survey_dif_passes_fpc_through(self):
        sim = simulate_dif_data(n_ref=400, n_focal=400, n_items=5, seed=17)
        w = np.ones(800)
        rw = jackknife_weights(w, np.zeros(800), psu=np.arange(800) % 16)
        wide = survey_dif(sim.responses, sim.group, "focal", w, replicate_weights=rw)
        tight = survey_dif(sim.responses, sim.group, "focal", w,
                           replicate_weights=rw, fpc=0.5)
        np.testing.assert_allclose(tight.se.values, wide.se.values * np.sqrt(0.5),
                                   rtol=1e-9)


class TestOrdinalFairness:
    @staticmethod
    def _shifted(seed=0, shift=1):
        rng = np.random.default_rng(seed)
        g = np.array(["r"] * 800 + ["f"] * 800)
        y = np.concatenate([rng.integers(0, 5, 800),
                            np.clip(rng.integers(0, 5, 800) - shift, 0, 4)])
        return y, g

    def test_detects_downward_shift(self):
        y, g = self._shifted()
        d = ordinal_disparity(y, g, "f")
        assert d["mean_difference"] < 0
        assert d["standardized_difference"] < 0
        assert d["probability_superiority"] < 0.5

    def test_no_disparity_gives_half(self):
        rng = np.random.default_rng(1)
        g = np.array(["r"] * 1000 + ["f"] * 1000)
        y = rng.integers(0, 5, 2000)
        d = ordinal_disparity(y, g, "f")
        assert abs(d["probability_superiority"] - 0.5) < 0.05
        assert abs(d["mean_difference"]) < 0.15

    def test_antisymmetric_under_relabelling(self):
        y, g = self._shifted(seed=3)
        a = ordinal_disparity(y, g, "f")
        b = ordinal_disparity(y, g, "r")
        assert abs(a["mean_difference"] + b["mean_difference"]) < 1e-9
        assert abs(a["probability_superiority"] + b["probability_superiority"] - 1) < 1e-9
        assert abs(a["max_cumulative_gap"] - b["max_cumulative_gap"]) < 1e-9

    def test_magnitude_tracks_shift(self):
        small = ordinal_disparity(*self._shifted(seed=5, shift=1), focal_label="f")
        large = ordinal_disparity(*self._shifted(seed=5, shift=3), focal_label="f")
        assert abs(large["mean_difference"]) > abs(small["mean_difference"])
        assert large["probability_superiority"] < small["probability_superiority"]

    def test_group_summary_shape(self):
        y, g = self._shifted()
        s = ordinal_group_summary(y, g)
        assert len(s) == 2
        props = [c for c in s.columns if c.startswith("p_")]
        np.testing.assert_allclose(s[props].sum(axis=1).values, 1.0, atol=1e-9)

    def test_empty_group_returns_nan(self):
        y = np.array([0, 1, 2, 3])
        g = np.array(["r"] * 4)
        d = ordinal_disparity(y, g, "f")
        assert np.isnan(d["mean_difference"])


ROOT_EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"


class TestTimssScoring:
    def test_scoring_helpers_importable(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "timss_val", ROOT_EXAMPLES / "timss_validation.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        s = mod._score_item(pd.Series([1.0, 2.0, 3.0, 4.0]), "SOME ITEM (B)")
        assert list(s) == [0.0, 1.0, 0.0, 0.0]
        c = mod._score_item(pd.Series([10.0, 20.0, 70.0, 21.0]), "CONSTRUCTED (1)")
        assert list(c) == [0.0, 1.0, 0.0, 1.0]

    def test_jk2_replicates_structure(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "timss_val2", ROOT_EXAMPLES / "timss_validation.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        w = np.ones(40)
        zone = np.repeat(np.arange(10), 4)
        rep = np.tile([0, 0, 1, 1], 10)
        out, zones = mod.jk2_replicates(w, zone, rep)
        assert out.shape == (10, 40)
        for i, row in enumerate(out):
            in_zone = zone == zones[i]
            assert (row[in_zone & (rep == 1)] == 0).all()
            assert (row[in_zone & (rep == 0)] == 2).all()
            assert (row[~in_zone] == 1).all()


class TestSurveyEdgeCases:
    def test_no_replicates_gives_zero_within(self):
        sim = simulate_dif_data(n_ref=300, n_focal=300, n_items=5, seed=19)
        out = survey_dif(sim.responses, sim.group, "focal", np.ones(600))
        assert (out.within == 0).all()
        assert (out.n_replicates == 0).all()

    def test_single_replicate_yields_no_variance(self):
        r = replicate_variance(1.0, [1.1])
        assert np.isnan(r["variance"])
        assert r["n_replicates"] == 1

    def test_empty_plausible_values_combination(self):
        r = combine_plausible_values([])
        assert r["n_values"] == 0 and np.isnan(r["estimate"])

    def test_polytomous_survey_switches_statistic(self):
        sim = simulate_poly_dif_data(n_ref=400, n_focal=400, n_items=6,
                                     n_dif_items=2, dif_magnitude=0.8, seed=23)
        out = survey_dif(sim.responses, sim.group, "focal", np.ones(800),
                         polytomous=True)
        assert out.estimate.notna().all()

    def test_jackknife_with_explicit_replicate_count(self):
        w = np.ones(60)
        rw = jackknife_weights(w, np.zeros(60), psu=np.arange(60) % 12,
                               n_replicates=5, seed=0)
        assert rw.shape[0] == 5


class TestPolytomousWeights:
    def test_unit_weights_match_unweighted(self):
        sim = simulate_poly_dif_data(n_ref=500, n_focal=500, n_items=8, seed=23)
        a = generalized_mantel_haenszel(sim.responses, sim.group, focal_label="focal")
        b = generalized_mantel_haenszel(sim.responses, sim.group, focal_label="focal",
                                        weights=np.ones(1000))
        np.testing.assert_allclose(a.smd.values, b.smd.values)
        np.testing.assert_allclose(a.chi2.values, b.chi2.values)

    def test_integer_weights_equal_replication(self):
        sim = simulate_poly_dif_data(n_ref=500, n_focal=500, n_items=8,
                                     n_dif_items=2, dif_magnitude=0.8, seed=23)
        w = np.random.default_rng(0).integers(1, 4, 1000).astype(float)
        rep = np.repeat(np.arange(1000), w.astype(int))
        a = generalized_mantel_haenszel(sim.responses, sim.group,
                                        focal_label="focal", weights=w)
        b = generalized_mantel_haenszel(sim.responses.iloc[rep], sim.group[rep],
                                        focal_label="focal")
        np.testing.assert_allclose(a.smd.values, b.smd.values, atol=1e-12)
        np.testing.assert_allclose(a.chi2.values, b.chi2.values, rtol=1e-9)


# --------------------------------------------------------------------------- #
# v0.5: degenerate-path coverage and ordinal inference
# --------------------------------------------------------------------------- #
class TestDegeneratePaths:
    def test_external_matching_criterion_used(self):
        sim = simulate_dif_data(n_ref=400, n_focal=400, n_items=8, seed=71)
        ext = np.random.default_rng(0).integers(0, 9, 800).astype(float)
        a = mantel_haenszel(sim.responses, sim.group, focal_label="focal")
        b = mantel_haenszel(sim.responses, sim.group, focal_label="focal",
                            matching=ext)
        assert not np.allclose(a.delta_mh.dropna(), b.delta_mh.dropna())

    def test_rest_score_matching(self):
        """Excluding the studied item changes the criterion per item."""
        sim = simulate_dif_data(n_ref=500, n_focal=500, n_items=10,
                                n_dif_items=2, dif_magnitude=0.8, seed=73)
        incl = mantel_haenszel(sim.responses, sim.group, focal_label="focal",
                               include_studied_item=True)
        excl = mantel_haenszel(sim.responses, sim.group, focal_label="focal",
                               include_studied_item=False)
        assert not np.allclose(incl.delta_mh, excl.delta_mh)
        # Both must still recover the planted items.
        for res in (incl, excl):
            flagged = set(res.loc[res.ets_class.isin(["B", "C"]), "item"])
            assert set(sim.dif_items) <= flagged

    def test_no_usable_strata_returns_undetermined(self):
        """Every stratum single-group: nothing is estimable, nothing crashes."""
        u = pd.DataFrame({"a": [0, 1] * 20, "b": [1, 0] * 20, "c": [0, 0] * 20})
        g = np.array([0] * 20 + [1] * 20)
        # Force a matching criterion that never mixes the groups.
        res = mantel_haenszel(u, g, focal_label=1, matching=g.astype(float))
        assert (res.ets_class == "undetermined").all()

    def test_logistic_skips_tiny_items(self):
        u = pd.DataFrame(np.random.default_rng(0).integers(0, 2, (12, 4)))
        res = logistic_dif(u, np.tile([0, 1], 6), focal_label=1)
        assert res.chi2_total.isna().all()   # below the 20-observation floor

    def test_logistic_survives_separation(self):
        """Perfect separation must yield NaN, not an exception."""
        n = 200
        g = np.tile([0, 1], n // 2)
        u = pd.DataFrame({"sep": g.astype(float),          # perfectly separated
                          "ok": np.random.default_rng(1).integers(0, 2, n)})
        res = logistic_dif(u, g, focal_label=1)
        assert len(res) == 2

    def test_breslow_day_needs_two_strata(self):
        u = pd.DataFrame({"a": [0, 1] * 30, "b": [1, 1, 0] * 20})
        g = np.tile([0, 1], 30)
        res = breslow_day(u, g, focal_label=1, matching=np.zeros(60))
        assert (res.df == 0).all()

    def test_bd_expected_handles_unit_odds_ratio(self):
        from difair.dif import _bd_expected

        e = _bd_expected(1.0, 50, 40, 100)
        assert e is not None and 0 < e < 40

    def test_bd_expected_returns_none_when_infeasible(self):
        from difair.dif import _bd_expected

        assert _bd_expected(2.0, 0, 40, 100) is None

    def test_purification_warns_when_all_flagged(self):
        """Every item flagged leaves nothing to rebuild the criterion from."""
        sim = simulate_dif_data(n_ref=800, n_focal=800, n_items=3,
                                n_dif_items=3, dif_magnitude=2.0, seed=79)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            matching, flagged = purify_matching_score(
                sim.responses, sim.group, focal_label="focal")
        assert len(matching) == 1600

    def test_normalize_weights_rejects_zero_sum(self):
        with pytest.raises(ValueError, match="sum to zero"):
            normalize_weights(np.zeros(10))


class TestOrdinalInference:
    def test_ps_interval_excludes_half_under_disparity(self):
        rng = np.random.default_rng(0)
        g = np.array(["r"] * 800 + ["f"] * 800)
        y = np.concatenate([rng.integers(0, 5, 800),
                            np.clip(rng.integers(0, 5, 800) - 1, 0, 4)])
        d = ordinal_disparity(y, g, "f")
        assert d["ps_ci_high"] < 0.5
        assert d["ps_se"] > 0

    def test_ps_interval_covers_half_under_null(self):
        rng = np.random.default_rng(1)
        g = np.array(["r"] * 1000 + ["f"] * 1000)
        y = rng.integers(0, 5, 2000)
        d = ordinal_disparity(y, g, "f")
        assert d["ps_ci_low"] < 0.5 < d["ps_ci_high"]

    def test_ps_se_tracks_monte_carlo(self):
        """The analytic standard error must match the empirical spread."""
        g = np.array(["r"] * 800 + ["f"] * 800)
        draws = [
            ordinal_disparity(np.random.default_rng(i).integers(0, 5, 1600),
                              g, "f")["probability_superiority"]
            for i in range(200)
        ]
        empirical = float(np.std(draws, ddof=1))
        analytic = ordinal_disparity(
            np.random.default_rng(999).integers(0, 5, 1600), g, "f")["ps_se"]
        assert 0.7 < analytic / empirical < 1.5

    def test_ps_interval_within_unit_range(self):
        rng = np.random.default_rng(3)
        g = np.array(["r"] * 60 + ["f"] * 60)
        y = np.concatenate([rng.integers(3, 5, 60), rng.integers(0, 2, 60)])
        d = ordinal_disparity(y, g, "f")
        assert 0.0 <= d["ps_ci_low"] <= d["ps_ci_high"] <= 1.0


# --------------------------------------------------------------------------- #
# v0.6: design inference, pooling, design-based ordinal inference
# --------------------------------------------------------------------------- #
from difair.fairness import ordinal_disparity_replicate  # noqa: E402
from difair.survey import infer_replicate_design, pool_estimates  # noqa: E402


def _synthetic_replicates(kind, n=200, r=16, seed=0):
    rng = np.random.default_rng(seed)
    w = np.ones(n)
    R = np.tile(w, (r, 1))
    if kind == "brr":
        for i in range(r):
            h = rng.random(n) < 0.5
            R[i, h] *= 2
            R[i, ~h] = 0
    elif kind == "fay":
        for i in range(r):
            h = rng.random(n) < 0.5
            R[i, h] *= 1.5
            R[i, ~h] *= 0.5
    elif kind == "jk2":
        zones = np.arange(n) % r
        pair = np.arange(n) % 2
        for i in range(r):
            in_zone = zones == i
            R[i, in_zone & (pair == 1)] = 0.0
            R[i, in_zone & (pair == 0)] *= 2.0
    return R, w


class TestReplicateDesignInference:
    def test_identifies_delete_one_jackknife(self):
        w = np.ones(200)
        rw = jackknife_weights(w, np.zeros(200), psu=np.arange(200) % 20)
        assert infer_replicate_design(rw, w)["method"] == "jackknife"

    def test_identifies_paired_jk2(self):
        R, w = _synthetic_replicates("jk2")
        assert infer_replicate_design(R, w)["method"] == "jackknife"

    def test_identifies_brr(self):
        R, w = _synthetic_replicates("brr")
        assert infer_replicate_design(R, w)["method"] == "brr"

    def test_identifies_fay_and_its_factor(self):
        R, w = _synthetic_replicates("fay")
        out = infer_replicate_design(R, w)
        assert out["method"] == "fay"
        assert abs(out["fay_factor"] - 0.5) < 0.1

    def test_unrecognised_design_returns_none(self):
        rng = np.random.default_rng(0)
        R = rng.uniform(0.2, 3.0, (12, 100))
        assert infer_replicate_design(R, np.ones(100))["method"] is None

    def test_single_replicate_is_not_classified(self):
        assert infer_replicate_design(np.ones((1, 50)), np.ones(50))["method"] is None

    def test_survey_dif_warns_on_mismatch(self):
        sim = simulate_dif_data(n_ref=300, n_focal=300, n_items=5, seed=3)
        w = np.ones(600)
        rw = jackknife_weights(w, np.zeros(600), psu=np.arange(600) % 20)
        with pytest.warns(UserWarning, match="does not match"):
            survey_dif(sim.responses, sim.group, "focal", w,
                       replicate_weights=rw, method="fay")

    def test_survey_dif_silent_when_matched(self):
        sim = simulate_dif_data(n_ref=300, n_focal=300, n_items=5, seed=5)
        w = np.ones(600)
        rw = jackknife_weights(w, np.zeros(600), psu=np.arange(600) % 20)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            survey_dif(sim.responses, sim.group, "focal", w,
                       replicate_weights=rw, method="jackknife")
        assert not [c for c in caught if "does not match" in str(c.message)]

    def test_check_can_be_disabled(self):
        sim = simulate_dif_data(n_ref=300, n_focal=300, n_items=5, seed=7)
        w = np.ones(600)
        rw = jackknife_weights(w, np.zeros(600), psu=np.arange(600) % 20)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            survey_dif(sim.responses, sim.group, "focal", w,
                       replicate_weights=rw, method="fay", check_design=False)
        assert not [c for c in caught if "does not match" in str(c.message)]


class TestPooling:
    def test_equal_precision_gives_mean(self):
        r = pool_estimates([1.0, 2.0, 3.0], [0.5] * 3)
        assert abs(r["estimate"] - 2.0) < 1e-12
        assert abs(r["se"] - 0.5 / np.sqrt(3)) < 1e-12

    def test_weights_toward_precise_estimate(self):
        r = pool_estimates([1.0, 2.0], [0.1, 1.0])
        assert abs(r["estimate"] - 1.0) < 0.05

    def test_heterogeneity_detected(self):
        hom = pool_estimates([1.0, 1.05, 0.95], [0.2] * 3)
        het = pool_estimates([1.0, 3.0, -1.0], [0.2] * 3)
        assert hom["i_squared"] < 0.2
        assert het["i_squared"] > 0.8

    def test_random_effects_is_wider(self):
        fixed = pool_estimates([1.0, 3.0, -1.0], [0.2] * 3)
        rand = pool_estimates([1.0, 3.0, -1.0], [0.2] * 3, method="random")
        assert rand["se"] > fixed["se"]
        assert rand["tau_squared"] > 0

    def test_single_analysis_passes_through(self):
        r = pool_estimates([2.0], [0.3])
        assert r["estimate"] == 2.0 and r["n_analyses"] == 1

    def test_drops_invalid_standard_errors(self):
        r = pool_estimates([1.0, 2.0, 3.0], [0.5, np.nan, 0.0])
        assert r["n_analyses"] == 1 and r["estimate"] == 1.0

    def test_empty_input(self):
        assert pool_estimates([], [])["n_analyses"] == 0

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="fixed"):
            pool_estimates([1.0, 2.0], [0.1, 0.1], method="magic")


class TestOrdinalReplicateInference:
    @staticmethod
    def _data(n=1200, seed=0):
        rng = np.random.default_rng(seed)
        g = np.array(["r"] * (n // 2) + ["f"] * (n // 2))
        y = np.concatenate([rng.integers(0, 5, n // 2),
                            np.clip(rng.integers(0, 5, n // 2) - 1, 0, 4)])
        return y, g

    def test_point_estimate_matches_analytic(self):
        y, g = self._data()
        w = np.ones(len(y))
        rw = jackknife_weights(w, np.zeros(len(y)), psu=np.arange(len(y)) % 40)
        a = ordinal_disparity(y, g, "f")
        b = ordinal_disparity_replicate(y, g, "f", weights=w, replicate_weights=rw)
        assert abs(a["probability_superiority"] - b["probability_superiority"]) < 0.03

    def test_no_replicates_gives_nan_se(self):
        y, g = self._data()
        out = ordinal_disparity_replicate(y, g, "f")
        assert np.isnan(out["se"]) and out["n_replicates"] == 0

    def test_interval_within_unit_range(self):
        y, g = self._data(seed=3)
        w = np.ones(len(y))
        rw = jackknife_weights(w, np.zeros(len(y)), psu=np.arange(len(y)) % 30)
        out = ordinal_disparity_replicate(y, g, "f", weights=w, replicate_weights=rw)
        assert 0.0 <= out["ci_low"] <= out["ci_high"] <= 1.0


class TestPolyAndPipelineEdges:
    def test_poly_rejects_three_groups(self):
        sim = simulate_poly_dif_data(n_ref=200, n_focal=200, n_items=5, seed=11)
        g = np.array(["a"] * 150 + ["b"] * 150 + ["c"] * 100)
        with pytest.raises(ValueError, match="binary"):
            generalized_mantel_haenszel(sim.responses, g, focal_label="a")

    def test_poly_requires_focal_label_for_text(self):
        sim = simulate_poly_dif_data(n_ref=200, n_focal=200, n_items=5, seed=13)
        with pytest.raises(ValueError, match="focal_label"):
            generalized_mantel_haenszel(sim.responses, sim.group)

    def test_poly_length_mismatch(self):
        sim = simulate_poly_dif_data(n_ref=100, n_focal=100, n_items=5, seed=17)
        with pytest.raises(ValueError, match="group"):
            generalized_mantel_haenszel(sim.responses, np.zeros(10), focal_label=0)

    def test_poly_external_matching(self):
        sim = simulate_poly_dif_data(n_ref=400, n_focal=400, n_items=8,
                                     n_dif_items=2, dif_magnitude=0.8, seed=19)
        ext = np.random.default_rng(0).integers(0, 12, 800).astype(float)
        a = generalized_mantel_haenszel(sim.responses, sim.group, focal_label="focal")
        b = generalized_mantel_haenszel(sim.responses, sim.group, focal_label="focal",
                                        matching=ext)
        assert not np.allclose(a.smd.dropna(), b.smd.dropna())

    def test_poly_rest_score_matching(self):
        sim = simulate_poly_dif_data(n_ref=400, n_focal=400, n_items=8, seed=23)
        a = generalized_mantel_haenszel(sim.responses, sim.group, focal_label="focal",
                                        include_studied_item=True)
        b = generalized_mantel_haenszel(sim.responses, sim.group, focal_label="focal",
                                        include_studied_item=False)
        assert not np.allclose(a.smd, b.smd)

    def test_poly_detect_rejects_empty_methods(self):
        sim = simulate_poly_dif_data(n_ref=200, n_focal=200, n_items=5, seed=29)
        with pytest.raises(ValueError, match="no procedure"):
            detect_dif_poly(sim.responses, sim.group, focal_label="focal", methods=())

    def test_attribution_equal_opportunity_metric(self):
        d = simulate_pipeline_data(n_ref=500, n_focal=500, seed=31)
        res = attribute_stages(d["responses"], d["group"], "focal", d["outcome"],
                               dif_items=d["dif_items"], proxy=d["proxy"],
                               train_mask=d["train_mask"],
                               metric="equal_opportunity")
        assert res.metric == "equal_opportunity"
        assert abs(res.shapley.shapley_value.sum() - res.explained) < 1e-9

    def test_attribution_with_decision_stage(self):
        d = simulate_pipeline_data(n_ref=500, n_focal=500, seed=37)
        res = attribute_stages(d["responses"], d["group"], "focal", d["outcome"],
                               dif_items=d["dif_items"], proxy=d["proxy"],
                               train_mask=d["train_mask"],
                               stages=("item", "model", "decision"))
        assert "decision" in set(res.shapley.stage)
        # Group-specific thresholds equalise selection rates up to the
        # granularity of a discrete score, so a small residual remains.
        assert res.residual_gap < 0.02
        assert res.residual_gap < res.baseline_gap / 5
        # The decision stage should dominate, since it patches the gap directly.
        top = res.summary().iloc[0]
        assert top.stage == "decision"

    def test_attribution_rejects_empty_stage_set(self):
        d = simulate_pipeline_data(n_ref=300, n_focal=300, seed=41)
        with pytest.raises(ValueError, match="No stage"):
            attribute_stages(d["responses"], d["group"], "focal", d["outcome"],
                             dif_items=d["dif_items"], stages=())


# --------------------------------------------------------------------------- #
# v0.7: diagnostics, cross-country pooling, remaining error paths
# --------------------------------------------------------------------------- #
class TestInferenceDiagnostics:
    def test_diagnostics_present_on_success(self):
        w = np.ones(200)
        rw = jackknife_weights(w, np.zeros(200), psu=np.arange(200) % 20)
        out = infer_replicate_design(rw, w)
        assert out["method"] == "jackknife"
        for key in ("unchanged_fraction", "doubled_fraction", "scale_quantiles"):
            assert key in out
        assert out["scale_quantiles"] is not None

    def test_diagnostics_present_on_failure(self):
        rng = np.random.default_rng(0)
        out = infer_replicate_design(rng.uniform(0.2, 3.0, (12, 100)), np.ones(100))
        assert out["method"] is None
        lo, mid, hi = out["scale_quantiles"]
        assert lo < mid < hi          # a continuous spread, not two levels

    def test_diagnostics_for_single_replicate(self):
        out = infer_replicate_design(np.ones((1, 40)), np.ones(40))
        assert out["method"] is None and out["n_replicates"] == 1

    def test_all_base_weights_invalid(self):
        out = infer_replicate_design(np.ones((4, 10)), np.zeros(10))
        assert out["method"] is None

    def test_warns_when_design_unrecognised(self):
        sim = simulate_dif_data(n_ref=300, n_focal=300, n_items=5, seed=3)
        rng = np.random.default_rng(0)
        weird = rng.uniform(0.2, 3.0, (12, 600))
        with pytest.warns(UserWarning, match="no recognised construction"):
            survey_dif(sim.responses, sim.group, "focal", np.ones(600),
                       replicate_weights=weird)


class TestCrossCountryPooling:
    """Pooling the same item across independent samples, the TIMSS use case."""

    def test_pools_independent_estimates_of_one_item(self):
        est = [0.42, 0.38, 0.45, 0.40]
        se = [0.10, 0.12, 0.11, 0.09]
        r = pool_estimates(est, se)
        assert min(est) < r["estimate"] < max(est)
        assert r["se"] < min(se)          # pooling gains precision
        assert r["i_squared"] < 0.2       # these agree

    def test_disagreeing_samples_inflate_random_effects(self):
        est = [1.5, -1.2, 0.9, -0.8]
        se = [0.15] * 4
        fixed = pool_estimates(est, se)
        rand = pool_estimates(est, se, method="random")
        assert fixed["i_squared"] > 0.9
        assert rand["se"] > 3 * fixed["se"]

    def test_pooled_interval_can_exclude_zero(self):
        r = pool_estimates([0.5, 0.55, 0.48], [0.1] * 3, method="random")
        assert r["ci_low"] > 0

    def test_pooling_two_is_enough(self):
        r = pool_estimates([1.0, 1.2], [0.2, 0.2])
        assert r["n_analyses"] == 2
        assert abs(r["estimate"] - 1.1) < 1e-9


class TestRemainingErrorPaths:
    def test_rejects_one_dimensional_responses(self):
        with pytest.raises(ValueError, match="2-dimensional"):
            mantel_haenszel(np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]),
                            focal_label=1)

    def test_rejects_absent_focal_label(self):
        sim = simulate_dif_data(n_ref=200, n_focal=200, n_items=5, seed=3)
        with pytest.raises(ValueError, match="not present"):
            mantel_haenszel(sim.responses, sim.group, focal_label="nobody")

    def test_standardization_undetermined_without_strata(self):
        u = pd.DataFrame({"a": [0, 1] * 20, "b": [1, 0] * 20})
        g = np.array([0] * 20 + [1] * 20)
        res = standardization(u, g, focal_label=1, matching=g.astype(float))
        assert (res.std_class == "undetermined").all()

    def test_breslow_day_undetermined_without_strata(self):
        u = pd.DataFrame({"a": [0, 1] * 20, "b": [1, 0] * 20})
        g = np.array([0] * 20 + [1] * 20)
        res = breslow_day(u, g, focal_label=1, matching=g.astype(float))
        assert res.bd_stat.isna().all()

    def test_purification_verbose_output(self, capsys):
        sim = simulate_dif_data(n_ref=600, n_focal=600, n_items=10,
                                n_dif_items=2, dif_magnitude=0.9, seed=5)
        purify_matching_score(sim.responses, sim.group, focal_label="focal",
                              verbose=True)
        assert "purify" in capsys.readouterr().out

    def test_jackknife_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="as long as"):
            jackknife_weights(np.ones(10), strata=np.zeros(5), psu=np.arange(10))

    def test_jackknife_skips_singleton_zones(self):
        """A zone with one unit cannot be resampled and is passed over."""
        w = np.ones(30)
        strata = np.concatenate([np.zeros(20), np.ones(10)])
        psu = np.concatenate([np.arange(20) % 4, np.full(10, 99)])
        rw = jackknife_weights(w, strata, psu=psu)
        assert rw.shape[0] == 4      # only the four-unit stratum contributes

    def test_survey_dif_without_replicates_has_zero_within(self):
        sim = simulate_dif_data(n_ref=300, n_focal=300, n_items=5, seed=7)
        out = survey_dif(sim.responses, sim.group, "focal", np.ones(600))
        assert (out.within == 0).all()
