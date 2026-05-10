"""Tests for the scenario YAML loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from selene.scenarios.loader import load_scenario
from selene.scenarios.modules.step_change import StepChangeAnomaly

SCENARIOS_DIR = Path(__file__).parent.parent.parent / "scenarios"
STEP_CHANGE_YAML = SCENARIOS_DIR / "test_step_change.yaml"


class TestLoadScenario:
    def test_loads_step_change_type(self):
        module = load_scenario(STEP_CHANGE_YAML)
        assert isinstance(module, StepChangeAnomaly)

    def test_name_from_yaml(self):
        module = load_scenario(STEP_CHANGE_YAML)
        assert module.name == "test_step_change"

    def test_affected_sensors_from_yaml(self):
        module = load_scenario(STEP_CHANGE_YAML)
        assert module.affected_sensors == ["tcs/temp-ams_in"]

    def test_ground_truth_populated(self):
        module = load_scenario(STEP_CHANGE_YAML)
        gt = module.get_ground_truth()
        assert gt.scenario_id == "test_step_change"
        assert "tcs/temp-ams_in" in gt.affected_sensors
        assert gt.start_time < gt.end_time

    def test_unknown_module_type_raises(self, tmp_path: Path):
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text(
            "module_type: does_not_exist\nname: x\ndescription: y\n"
            "affected_sensors: []\nstart_time: '2020-01-01T00:00:00+00:00'\n"
            "end_time: '2020-01-01T01:00:00+00:00'\noffset: 1.0\n"
        )
        with pytest.raises(KeyError, match="does_not_exist"):
            load_scenario(yaml_path)
