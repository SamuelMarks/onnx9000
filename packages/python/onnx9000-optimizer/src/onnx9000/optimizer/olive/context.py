"""Provides the PassContext tracking intermediate optimization states."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PassContext:
    """Tracks intermediate optimization states."""

    state: dict[str, Any] = field(default_factory=dict)
