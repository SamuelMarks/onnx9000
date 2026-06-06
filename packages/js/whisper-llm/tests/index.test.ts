import { describe, it, expect } from "vitest";
import { WhisperLlm } from "../src/index.js";

describe("WhisperLlm", () => {
  it("should transcribe", () => {
    const w = new WhisperLlm();
    expect(w.transcribe("test")).toContain("[Whisper-LLM]");
    expect(() => w.transcribe("")).toThrow();
  });
});
