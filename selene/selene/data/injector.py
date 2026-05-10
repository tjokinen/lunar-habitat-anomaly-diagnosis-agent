"""ScenarioInjector — wraps a TelemetrySource and applies AnomalyModule transforms."""

from __future__ import annotations

from typing import AsyncIterator

from selene.core.interfaces import AnomalyModule, SensorMetadata, TelemetryFrame, TelemetrySource


class ScenarioInjector:
    """Wraps any TelemetrySource and applies registered AnomalyModule instances.

    Implements TelemetrySource itself so downstream code is unaware of injection.
    Modules are applied in registration order (deterministic).
    """

    def __init__(self, source: TelemetrySource, modules: list[AnomalyModule]) -> None:
        self._source = source
        self._modules = list(modules)

    async def stream(self) -> AsyncIterator[TelemetryFrame]:
        async for frame in self._source.stream():
            for module in self._modules:
                frame = module.transform(frame)
            yield frame

    def get_metadata(self) -> SensorMetadata:
        return self._source.get_metadata()
