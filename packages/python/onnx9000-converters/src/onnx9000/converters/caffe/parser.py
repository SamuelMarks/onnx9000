"""Caffe prototxt parser."""

import re
from typing import Any


def parse_prototxt(content: str) -> dict[str, Any]:
    """Parse a Caffe .prototxt file into a dictionary.

    Args:
        content (str): The string content of the .prototxt file.

    Returns:
        Dict[str, Any]: A dictionary representing the Caffe network.
    """
    # Remove comments
    content = re.sub(r"#.*", "", content)

    tokens = re.findall(
        r'([A-Za-z0-9_]+)\s*:\s*"([^"]*)"|([A-Za-z0-9_]+)\s*:\s*([^\{\s]+)|([A-Za-z0-9_]+)\s*\{|\}',
        content,
    )

    stack = []
    current: dict[str, Any] = {}

    for string_k, string_v, num_k, num_v, block_k in tokens:
        if string_k:
            if string_k not in current:
                current[string_k] = []
            current[string_k].append(string_v)
        elif num_k:
            val = float(num_v) if "." in num_v or "e" in num_v.lower() else int(num_v)
            if num_k not in current:
                current[num_k] = []
            current[num_k].append(val)
        elif block_k:
            stack.append((block_k, current))
            current = {}
        else:  # '}'
            if stack:
                parent_k, parent_dict = stack.pop()
                if parent_k not in parent_dict:
                    parent_dict[parent_k] = []
                parent_dict[parent_k].append(current)
                current = parent_dict

    return current
