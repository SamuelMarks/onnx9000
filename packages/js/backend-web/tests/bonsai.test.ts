import { describe, expect, it } from "vitest";
import {
  FusedLayerNormWGSL,
  FusedRoPEWGSL,
} from "../src/providers/webgpu/shaders/bonsai.js";

describe("bonsai shaders", () => {
  it("should export wgsl", () => {
    expect(FusedLayerNormWGSL).toContain("@compute");
    expect(FusedRoPEWGSL).toContain("@compute");
  });
});
