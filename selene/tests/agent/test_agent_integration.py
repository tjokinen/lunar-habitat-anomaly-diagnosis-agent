"""Integration test for ReasoningAgent against a real vLLM endpoint.

Skipped by default. Runs only when ``SELENE_LLM_BASE_URL`` is set, the headline
thermal-leak scenario from step 2.8 is loadable, and the KB has been populated.

The test reproduces the headline diagnostic flow end-to-end:

1. Replay enough EDEN ISS frames into a TelemetryStore that the leak signature
   has accumulated.
2. Fire a synthetic AnomalyEvent on the affected pressure sensor.
3. Run ``agent.investigate(...)`` against the live vLLM model.
4. Assert that ``Diagnosis.matched_failure_modes`` contains
   ``iss_p1_eatcs_leak_2011``.

Stays skipped until step 2.8 lands; the headline scenario module and replay
fixture it depends on do not exist yet.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.slow


@pytest.mark.skipif(
    "SELENE_LLM_BASE_URL" not in os.environ,
    reason="Set SELENE_LLM_BASE_URL to run the live-vLLM integration test.",
)
@pytest.mark.skip(
    reason=(
        "Depends on the headline thermal-leak scenario module (step 2.8). "
        "Re-enable once selene/scenarios/modules/thermal_leak.py exists."
    )
)
def test_headline_thermal_leak_diagnosis_against_live_vllm() -> None:
    """Headline integration: replayer + thermal-leak scenario + agent + vLLM.

    Asserts ``Diagnosis.matched_failure_modes`` contains the ISS P1 EATCS leak
    KB entry. The exact confidence and supporting evidence depend on the model
    and are not asserted here.
    """
    # Implementation deferred to step 2.8.
    raise NotImplementedError
