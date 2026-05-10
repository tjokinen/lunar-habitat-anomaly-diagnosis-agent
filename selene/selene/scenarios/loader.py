"""YAML-based scenario loader."""

from __future__ import annotations

from pathlib import Path

import yaml

# Importing modules package triggers all @register_module decorators
import selene.scenarios.modules  # noqa: F401
from selene.core.interfaces import AnomalyModule
from selene.scenarios.registry import get_module_class


def load_scenario(yaml_path: str | Path) -> AnomalyModule:
    """Load an AnomalyModule from a YAML config file.

    The YAML must contain a ``module_type`` key that matches a registered
    module class.  All remaining keys are passed as kwargs to the class
    constructor (validated via the class's Pydantic config model).
    """
    path = Path(yaml_path)
    with path.open() as f:
        config: dict = yaml.safe_load(f)

    module_type: str = config.pop("module_type")
    cls = get_module_class(module_type)
    return cls(**config)  # type: ignore[return-value]
