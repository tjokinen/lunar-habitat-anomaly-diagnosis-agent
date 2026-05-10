"""Symptom-based failure mode matcher.

Structured retrieval over typed KB fields — no embedding similarity.
"""

from __future__ import annotations

from collections import Counter

from selene.knowledge.models import FailureMode, Signature


def match_failure_modes(
    symptoms: list[Signature],
    kb: dict[str, FailureMode],
) -> list[tuple[FailureMode, float]]:
    """Return KB entries ranked by symptom overlap, highest score first.

    Score formula
    -------------
    For each KB entry the score is:

        score = weighted_match(primary) + 0.5 * weighted_match(secondary)

    where ``weighted_match(sigs)`` is the fraction of *sigs* that are matched
    by at least one symptom, with each signature weighted by its *distinctiveness*
    — the inverse of how many KB entries share the same (sensor_pattern,
    pattern_type) pair.  Rare patterns score higher when matched; common patterns
    contribute less.

    A symptom matches a KB signature when:
    - ``sensor_pattern``: the strings share a common token (partial overlap)
    - ``pattern_type``: exact match
    - ``direction``: exact match, or either side is ``"either"``
    """
    if not kb or not symptoms:
        return []

    # Pre-compute distinctiveness weights: inverse frequency of (sensor_pattern, pattern_type)
    pattern_counts: Counter[tuple[str, str]] = Counter()
    for fm in kb.values():
        for sig in fm.primary_signature + fm.secondary_signature:
            pattern_counts[(sig.sensor_pattern, sig.pattern_type)] += 1

    def _distinctiveness(sig: Signature) -> float:
        count = pattern_counts.get((sig.sensor_pattern, sig.pattern_type), 1)
        return 1.0 / count

    def _symptom_matches(symptom: Signature, kb_sig: Signature) -> bool:
        # pattern_type must match exactly
        if symptom.pattern_type != kb_sig.pattern_type:
            return False
        # direction: "either" on either side is a wildcard
        if symptom.direction != "either" and kb_sig.direction != "either":
            if symptom.direction != kb_sig.direction:
                return False
        # sensor_pattern: accept partial overlap via shared tokens
        sym_tokens = set(symptom.sensor_pattern.replace("-", "_").split("_"))
        kb_tokens = set(kb_sig.sensor_pattern.replace("-", "_").split("_"))
        if not sym_tokens & kb_tokens:
            return False
        return True

    def _weighted_match(signatures: list[Signature]) -> float:
        if not signatures:
            return 0.0
        total_weight = sum(_distinctiveness(s) for s in signatures)
        if total_weight == 0:
            return 0.0
        matched_weight = 0.0
        for kb_sig in signatures:
            if any(_symptom_matches(sym, kb_sig) for sym in symptoms):
                matched_weight += _distinctiveness(kb_sig)
        return matched_weight / total_weight

    results: list[tuple[FailureMode, float]] = []
    for fm in kb.values():
        score = _weighted_match(fm.primary_signature) + 0.5 * _weighted_match(
            fm.secondary_signature
        )
        results.append((fm, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
