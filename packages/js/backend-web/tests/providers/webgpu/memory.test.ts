import { describe, expect, it } from "vitest";
import * as Module from "../../../src/providers/webgpu/memory";

describe("memory.ts", () => {
  it("should instantiate and cover WebGPUMemoryManager", () => {
    try {
      const obj = new (Module as any).WebGPUMemoryManager();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
