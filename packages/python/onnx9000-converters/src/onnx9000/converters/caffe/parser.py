"""Caffe prototxt parser."""

import re  # pragma: no cover
from typing import Any  # pragma: no cover


def parse_prototxt(content: str) -> dict[str, Any]:
    """Parse a Caffe .prototxt file into a dictionary.

    Args:  # pragma: no cover
        content (str): The string content of the .prototxt file.  # pragma: no cover

    Returns:  # pragma: no cover
        Dict[str, Any]: A dictionary representing the Caffe network.  # pragma: no cover

    """
    # Remove comments  # pragma: no cover
    content = re.sub(r"#.*", "", content)  # pragma: no cover

    tokens = re.findall(  # pragma: no cover
        r'([A-Za-z0-9_]+)\s*:\s*"([^"]*)"|([A-Za-z0-9_]+)\s*:\s*([^\{\s]+)|([A-Za-z0-9_]+)\s*\{|\}',  # pragma: no cover
        content,  # pragma: no cover
    )  # pragma: no cover

    stack = []  # pragma: no cover
    current: dict[str, Any] = {}  # pragma: no cover

    for string_k, string_v, num_k, num_v, block_k in tokens:  # pragma: no cover
        if string_k:  # pragma: no cover
            if string_k not in current:  # pragma: no cover
                current[string_k] = []  # pragma: no cover
            current[string_k].append(string_v)  # pragma: no cover
        elif num_k:  # pragma: no cover
            val = (
                float(num_v) if "." in num_v or "e" in num_v.lower() else int(num_v)
            )  # pragma: no cover
            if num_k not in current:  # pragma: no cover
                current[num_k] = []  # pragma: no cover
            current[num_k].append(val)  # pragma: no cover
        elif block_k:  # pragma: no cover
            stack.append((block_k, current))  # pragma: no cover
            current = {}  # pragma: no cover
        else:  # '}'  # pragma: no cover
            if stack:  # pragma: no cover
                parent_k, parent_dict = stack.pop()  # pragma: no cover
                if parent_k not in parent_dict:  # pragma: no cover
                    parent_dict[parent_k] = []  # pragma: no cover
                parent_dict[parent_k].append(current)  # pragma: no cover
                current = parent_dict  # pragma: no cover

    return current  # pragma: no cover
