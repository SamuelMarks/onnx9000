import { describe, it, expect } from "vitest";
import { WebGPUManager } from "../src/webgpu.js";

describe("WebGPUManager", () => {
  it("should init", async () => {
    const mgr = new WebGPUManager();
    await mgr.init();
    expect(mgr.fallbackToWasm).toBe(true);
  });
});
