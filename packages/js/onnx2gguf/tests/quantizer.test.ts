import { describe, expect, it } from "vitest";
import {
  f32ToF16,
  quantizeQ4_0,
  quantizeQ4_1,
  quantizeQ8_0,
} from "../src/quantizer.js";

describe("quantizer", () => {
  it("should quantize", () => {
    const f32 = new Float32Array(32).fill(1.0);
    const u8 = new Uint8Array(f32.buffer);

    expect(f32ToF16(u8).length).toBe(64);
    expect(quantizeQ4_0(u8).length).toBe(18);
    expect(quantizeQ4_1(u8).length).toBe(20);
    expect(quantizeQ8_0(u8).length).toBe(34);
  });
});
