def run_model(model_str: str) -> str:
    if not model_str:
        raise ValueError("Invalid model string")
    return f"[LLaMA-Web] processing {model_str}"
