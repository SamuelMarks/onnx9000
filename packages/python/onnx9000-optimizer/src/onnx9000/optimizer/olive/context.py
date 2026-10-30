"""Provides the PassContext tracking intermediate optimization states."""

from dataclasses import dataclass, field


@dataclass
class PassContext:
    """Tracks intermediate optimization states."""

    state: dict[str, int | float | str | bool | bytes | list | dict | tuple | None] = field(
        default_factory=dict
    )
