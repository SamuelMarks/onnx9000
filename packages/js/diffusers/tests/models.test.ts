import { describe, it, expect } from "vitest";
import { AutoencoderKL, UNet2DConditionModel } from "../src/models.js";

describe("diffusers models", () => {
  it("should AutoencoderKL", () => {
    const a = new AutoencoderKL();
    expect(a.encode(new Float32Array([1]))[0]).toBeCloseTo(0.18215);
    expect(a.decode(new Float32Array([0.18215]))[0]).toBeCloseTo(1);
  });

  it("should UNet", () => {
    const u = new UNet2DConditionModel();
    expect(
      u.call(new Float32Array([1]), 10, new Float32Array())[0],
    ).toBeCloseTo(0.9);
  });
});
