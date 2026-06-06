/**
 * @fileoverview index.ts
 * Provides index functionality for the whisper-llm package.
 */
export class WhisperLlm {
  public transcribe(audioString: string): string {
    if (!audioString) {
      throw new Error("Invalid audio string");
    }
    return `[Whisper-LLM] transcribed ${audioString}`;
  }
}
