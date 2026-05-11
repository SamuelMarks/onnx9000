"""MXNet symbol parser."""

import json
from typing import Any


def parse_symbol(content: str) -> dict[str, Any]:
    """Parse MXNet -symbol.json.

    Args:
        content: JSON string content.

    Returns:
        Dict: Parsed JSON dictionary.
    """
    return json.loads(content)
