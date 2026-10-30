"""MXNet symbol parser."""

import json


def parse_symbol(
    content: str,
) -> dict[str, int | float | str | bool | bytes | list | dict | tuple | None]:
    """Parse MXNet -symbol.json.

    Args:
        content: JSON string content.

    Returns:
        Dict: Parsed JSON dictionary.

    """
    return json.loads(content)
