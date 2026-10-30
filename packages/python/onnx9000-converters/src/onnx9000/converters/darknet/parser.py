"""Darknet configuration parser."""


def parse_cfg(
    content: str,
) -> list[dict[str, int | float | str | bool | bytes | list | dict | tuple | None]]:
    """Parse a Darknet .cfg file into a list of layer configurations.

    Args:
        content (str): The string content of the .cfg file.

    Returns:
        List[dict[str, Union[int, float, str, bool]]]: A list of dictionaries representing the layers.

    """
    lines = content.split("\n")
    blocks = []
    block = None

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("["):
            if block is not None:
                blocks.append(block)
            block = {"type": line[1:-1].strip()}
        elif "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if block is not None:
                block[key] = value

    if block is not None:
        blocks.append(block)

    return blocks
