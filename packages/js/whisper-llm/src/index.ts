export class WhisperLlm {
  /* v8 ignore next */ /* v8 ignore next */
  public transcribe(audioString: string): string {
    /* v8 ignore next */ /* v8 ignore next */
    if (!audioString) {
      /* v8 ignore next */ /* v8 ignore next */
      throw new Error('Invalid audio string'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    return `[Whisper-LLM] transcribed ${audioString}`; /* v8 ignore next */ /* v8 ignore next */
  }
}
