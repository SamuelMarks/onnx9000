import { describe, it, expect } from "vitest";
import { WebGPUProvider, MatMulWebGPU } from "../src/providers/webgpu/index.js";

describe("WebGPUProvider", () => {
  it("should execute matmul fallback", () => {
    const op = new MatMulWebGPU();
    const res = op.execute([{} as any, { format: "dense" } as any], {});
    expect(res.length).toBe(0);
  });

  it("should initialize", async () => {
    const prov = new WebGPUProvider();
    (global as any).navigator = {
      gpu: {
        requestAdapter: async () => ({ requestDevice: async () => ({}) }),
      },
    };
    await prov.initialize();
    expect((prov as any).device).toBeDefined();
  });
});
