def plan(model_str: str) -> str:
    if not model_str:
        raise ValueError("Invalid model string")
    return f"[Arena] planner processed {model_str}"
