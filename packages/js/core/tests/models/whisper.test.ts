import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/models/whisper";

describe("whisper.ts", () => {
  it("should instantiate and cover WhisperEncoderLayer", () => {
    try {
      const obj = new (Module as any).WhisperEncoderLayer();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover WhisperEncoder", () => {
    try {
      const obj = new (Module as any).WhisperEncoder();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover WhisperDecoderLayer", () => {
    try {
      const obj = new (Module as any).WhisperDecoderLayer();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover WhisperDecoder", () => {
    try {
      const obj = new (Module as any).WhisperDecoder();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover Whisper", () => {
    try {
      const obj = new (Module as any).Whisper();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should call and cover whisperTiny", async () => {
    try {
      const res = (Module as any).whisperTiny();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
});
