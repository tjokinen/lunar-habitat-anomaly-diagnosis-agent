"""Pydantic models for the failure mode knowledge base."""

from __future__ import annotations

from pydantic import BaseModel


class Signature(BaseModel):
    sensor_pattern: str
    pattern_type: str        # "slow_drift" | "step_change" | "oscillation" | "threshold_breach"
    direction: str           # "increasing" | "decreasing" | "either"
    time_scale: str          # "minutes" | "hours" | "days" | "weeks"
    correlation_with: list[str] = []


class Citation(BaseModel):
    source_type: str         # "NTRS" | "ESA" | "paper" | "ISS_daily" | "EDEN_ISS_dataset"
    identifier: str
    title: str
    url: str | None = None


class FailureMode(BaseModel):
    id: str
    name: str
    affected_subsystems: list[str]
    primary_signature: list[Signature]
    secondary_signature: list[Signature] = []
    typical_onset: str
    distinguishing_features: list[str]
    differential_diagnosis: list[str] = []
    historical_context: str
    citations: list[Citation]
    typical_response: str
