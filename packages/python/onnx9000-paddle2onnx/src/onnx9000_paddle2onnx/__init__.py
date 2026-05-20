def convert(model_str: str) -> str:
    if not model_str:
        raise ValueError("Invalid model string")
    return f"[ONNX-IR] from {model_str}"
