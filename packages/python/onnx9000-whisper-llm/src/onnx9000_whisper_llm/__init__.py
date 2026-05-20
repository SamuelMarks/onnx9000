def transcribe(audio_str: str) -> str:
    if not audio_str:
        raise ValueError("Invalid audio string")
    return f"[Whisper-LLM] transcribed {audio_str}"
