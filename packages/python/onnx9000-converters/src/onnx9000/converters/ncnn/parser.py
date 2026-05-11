"""NCNN param parser."""

from typing import Any


def parse_param(content: str) -> dict[str, Any]:
    """Parse NCNN .param file.

    Args:
        content (str): The string content of the .param file.

    Returns:
        Dict[str, Any]: Parsed model info including layers.
    """
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    if not lines:
        raise ValueError("Empty NCNN param file")

    magic = int(lines[0])
    if magic != 7767517:
        raise ValueError(f"Invalid NCNN magic number: {magic}")

    counts = lines[1].split()
    layer_count = int(counts[0])
    blob_count = int(counts[1])

    layers = []
    for line in lines[2:]:
        parts = line.split()
        if not parts:
            continue

        layer_type = parts[0]
        layer_name = parts[1]
        bottom_count = int(parts[2])
        top_count = int(parts[3])

        idx = 4
        bottom_names = parts[idx : idx + bottom_count]
        idx += bottom_count

        top_names = parts[idx : idx + top_count]
        idx += top_count

        params = {}
        for p in parts[idx:]:
            if "=" in p:
                k, v = p.split("=", 1)
                try:
                    if "." in v or "e" in v.lower():
                        params[int(k)] = float(v)
                    else:
                        params[int(k)] = int(v)
                except ValueError:
                    params[int(k)] = v

        layers.append(
            {
                "type": layer_type,
                "name": layer_name,
                "bottoms": bottom_names,
                "tops": top_names,
                "params": params,
            }
        )

    return {"magic": magic, "layer_count": layer_count, "blob_count": blob_count, "layers": layers}
