import { describe, expect, it } from "vitest";
import {
  GatherNDWGSL,
  ScatterNDWGSL,
} from "../src/providers/webgpu/shaders/scatter_nd.js";

describe("scatter_nd shaders", () => {
  it("should export wgsl", () => {
    expect(GatherNDWGSL).toContain("@compute");
    expect(ScatterNDWGSL).toContain("@compute");
  });
});
