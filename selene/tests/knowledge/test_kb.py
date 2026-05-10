"""Tests for knowledge base loader and symptom matcher."""

from __future__ import annotations

from pathlib import Path

import pytest

from selene.knowledge.loader import load_kb
from selene.knowledge.matcher import match_failure_modes
from selene.knowledge.models import Citation, FailureMode, Signature

KB_DIR = Path(__file__).parent.parent.parent / "knowledge"

# ── Loader ────────────────────────────────────────────────────────────────────

class TestKBLoader:
    def test_loads_all_five_entries(self):
        kb = load_kb(KB_DIR)
        assert len(kb) == 5

    def test_ids_match_filenames(self):
        kb = load_kb(KB_DIR)
        for entry_id, fm in kb.items():
            assert fm.id == entry_id

    def test_headline_entry_present(self):
        kb = load_kb(KB_DIR)
        assert "iss_p1_eatcs_leak_2011" in kb

    def test_headline_entry_schema(self):
        kb = load_kb(KB_DIR)
        fm = kb["iss_p1_eatcs_leak_2011"]
        assert isinstance(fm, FailureMode)
        assert fm.affected_subsystems
        assert fm.primary_signature
        assert fm.citations
        assert isinstance(fm.citations[0], Citation)
        assert fm.citations[0].source_type == "NTRS"

    def test_native_entries_present(self):
        kb = load_kb(KB_DIR)
        assert "eden_iss_pump_degradation" in kb
        assert "eden_iss_co2_scrubber_drift" in kb
        assert "eden_iss_illumination_degradation" in kb

    def test_all_entries_pass_schema_validation(self):
        kb = load_kb(KB_DIR)
        for fm in kb.values():
            assert fm.id
            assert fm.name
            assert fm.primary_signature  # non-empty
            assert fm.citations          # non-empty
            for sig in fm.primary_signature + fm.secondary_signature:
                assert sig.pattern_type in {
                    "slow_drift", "step_change", "oscillation", "threshold_breach"
                }
                assert sig.direction in {"increasing", "decreasing", "either"}
                assert sig.time_scale in {"minutes", "hours", "days", "weeks"}

    def test_missing_directory_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_kb(tmp_path / "nonexistent")

    def test_bad_yaml_skipped_gracefully(self, tmp_path: Path):
        # Copy a valid entry then add a malformed one
        import shutil
        shutil.copy(KB_DIR / "iss_p1_eatcs_leak_2011.yaml", tmp_path / "iss_p1_eatcs_leak_2011.yaml")
        (tmp_path / "bad_entry.yaml").write_text("id: bad\nthis is not valid yaml: [unclosed")
        kb = load_kb(tmp_path)
        # The bad file is skipped; the valid one loads
        assert "iss_p1_eatcs_leak_2011" in kb
        assert "bad_entry" not in kb


# ── Matcher ───────────────────────────────────────────────────────────────────

def _iss_leak_symptoms() -> list[Signature]:
    """Symptoms that directly match the ISS P1 EATCS leak primary signatures."""
    return [
        Signature(
            sensor_pattern="thermal_loop_pressure",
            pattern_type="slow_drift",
            direction="decreasing",
            time_scale="weeks",
        ),
        Signature(
            sensor_pattern="thermal_loop_temperature",
            pattern_type="slow_drift",
            direction="increasing",
            time_scale="hours",
        ),
    ]


def _pump_symptoms() -> list[Signature]:
    """Symptoms aligned with the NDS pump degradation entry."""
    return [
        Signature(
            sensor_pattern="nutrient_loop_pressure",
            pattern_type="oscillation",
            direction="either",
            time_scale="hours",
        ),
        Signature(
            sensor_pattern="nutrient_electrical_conductivity",
            pattern_type="oscillation",
            direction="either",
            time_scale="hours",
        ),
    ]


class TestMatcher:
    def test_iss_leak_symptoms_rank_first(self):
        kb = load_kb(KB_DIR)
        results = match_failure_modes(_iss_leak_symptoms(), kb)
        assert results[0][0].id == "iss_p1_eatcs_leak_2011"

    def test_pump_symptoms_rank_pump_entry_first(self):
        kb = load_kb(KB_DIR)
        results = match_failure_modes(_pump_symptoms(), kb)
        assert results[0][0].id == "eden_iss_pump_degradation"

    def test_returns_all_entries(self):
        kb = load_kb(KB_DIR)
        results = match_failure_modes(_iss_leak_symptoms(), kb)
        assert len(results) == len(kb)

    def test_scores_descending(self):
        kb = load_kb(KB_DIR)
        results = match_failure_modes(_iss_leak_symptoms(), kb)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_no_match_yields_low_scores(self):
        kb = load_kb(KB_DIR)
        unrelated = [
            Signature(
                sensor_pattern="xray_flux_gamma",
                pattern_type="step_change",
                direction="increasing",
                time_scale="minutes",
            )
        ]
        results = match_failure_modes(unrelated, kb)
        # All scores should be very low (no token overlap with any KB entry)
        assert all(score < 0.2 for _, score in results)

    def test_empty_symptoms_returns_all_zero(self):
        kb = load_kb(KB_DIR)
        results = match_failure_modes([], kb)
        assert results == []

    def test_empty_kb_returns_empty(self):
        results = match_failure_modes(_iss_leak_symptoms(), {})
        assert results == []

    def test_direction_either_wildcard(self):
        """A symptom with direction='either' should match kb sigs of any direction."""
        kb = load_kb(KB_DIR)
        wildcard = [
            Signature(
                sensor_pattern="thermal_loop_pressure",
                pattern_type="slow_drift",
                direction="either",
                time_scale="weeks",
            )
        ]
        results = match_failure_modes(wildcard, kb)
        iss_score = next(s for fm, s in results if fm.id == "iss_p1_eatcs_leak_2011")
        assert iss_score > 0

    def test_top_match_score_positive(self):
        kb = load_kb(KB_DIR)
        results = match_failure_modes(_iss_leak_symptoms(), kb)
        assert results[0][1] > 0
