import { describe, it, expect, vi } from "vitest";
import * as Module from "../../../../src/providers/webgpu/shaders/keras_layer_norm";

describe("keras_layer_norm.ts", () => {
  it("should load module", () => {
    expect(Module).toBeDefined();
  });
});
