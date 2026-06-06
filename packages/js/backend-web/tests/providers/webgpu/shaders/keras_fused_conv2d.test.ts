import { describe, it, expect, vi } from "vitest";
import * as Module from "../../../../src/providers/webgpu/shaders/keras_fused_conv2d";

describe("keras_fused_conv2d.ts", () => {
  it("should load module", () => {
    expect(Module).toBeDefined();
  });
});
