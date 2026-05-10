"""Load and validate failure mode YAML files from the knowledge/ directory."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from selene.knowledge.models import FailureMode

logger = logging.getLogger(__name__)

# Default path is relative to the repo root (pyproject.toml lives there).
_DEFAULT_KB_DIR = Path(__file__).parent.parent.parent / "knowledge"


def load_kb(kb_dir: str | Path | None = None) -> dict[str, FailureMode]:
    """Load all ``*.yaml`` files from *kb_dir* and return a dict keyed by entry id.

    Each file is validated against the :class:`FailureMode` schema.  Files that
    fail validation are skipped with a warning so a single bad entry never
    prevents the rest of the KB from loading.
    """
    directory = Path(kb_dir) if kb_dir is not None else _DEFAULT_KB_DIR

    if not directory.is_dir():
        raise FileNotFoundError(f"Knowledge base directory not found: {directory}")

    kb: dict[str, FailureMode] = {}
    for path in sorted(directory.glob("*.yaml")):
        try:
            with path.open() as f:
                raw = yaml.safe_load(f)
            entry = FailureMode.model_validate(raw)
            if entry.id != path.stem:
                logger.warning(
                    "KB entry id %r does not match filename %r — using id from file content",
                    entry.id,
                    path.stem,
                )
            kb[entry.id] = entry
        except Exception as exc:
            logger.warning("Failed to load KB entry %s: %s", path.name, exc)

    logger.debug("Loaded %d KB entries from %s", len(kb), directory)
    return kb
