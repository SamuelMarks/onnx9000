"""Caffe prototxt parser."""

import re


def parse_prototxt(
    content: str,
) -> dict[str, int | float | str | bool | bytes | list | dict | tuple | None]:
    """Parse a Caffe .prototxt file into a dictionary.

    Args:
        content: The string content of the .prototxt file.

    Returns:
        A dictionary representing the Caffe network.

    """
    content = re.sub(r"#.*", "", content)

    tokens = re.findall(
        r'([A-Za-z0-9_]+)\s*:\s*"([^"]*)"|([A-Za-z0-9_]+)\s*:\s*([^\{\s]+)|([A-Za-z0-9_]+)\s*\{|\}',
        content,
    )

    stack = []
    current: dict[str, int | float | str | bool | bytes | list | dict | tuple | None] = {}

    for string_k, string_v, num_k, num_v, block_k in tokens:
        if string_k:
            if string_k not in current:
                current[string_k] = []
            current[string_k].append(string_v)
        elif num_k:
            if num_v.lower() == "true":
                val: int | float | str | bool = True
            elif num_v.lower() == "false":
                val = False
            else:
                try:
                    val = float(num_v) if "." in num_v or "e" in num_v.lower() else int(num_v)
                except ValueError:
                    val = num_v
            if num_k not in current:
                current[num_k] = []
            current[num_k].append(val)
        elif block_k:
            stack.append((block_k, current))
            current = {}
        elif stack:
            parent_k, parent_dict = stack.pop()
            if parent_k not in parent_dict:
                parent_dict[parent_k] = []
            parent_dict[parent_k].append(current)
            current = parent_dict

    return current
