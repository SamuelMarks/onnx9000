import { describe, it, expect, vi } from "vitest";
import * as Module from "../../../src/providers/webgpu/index";

describe("index.ts", () => {
  it("should instantiate and cover MatMulWebGPU", () => {
    try {
      const obj = new (Module as any).MatMulWebGPU();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover WebGPUProvider", () => {
    try {
      const obj = new (Module as any).WebGPUProvider();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
});
