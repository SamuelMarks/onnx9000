import { describe, expect, it, vi } from "vitest";
import {
  fetchHubFile,
  globalProgressBarConfig,
  PyTorchPCG,
  parseModelIndex,
  rand,
  randn,
  setProgressBarConfig,
} from "../src/utils.js";

describe("diffusers utils", () => {
  it("should generate pcg", () => {
    const pcg = new PyTorchPCG(42);
    expect(pcg.nextFloat()).toBeLessThanOrEqual(1.0);

    expect(rand([2], pcg).length).toBe(2);
    expect(randn([2], pcg).length).toBe(2);
  });

  it("should fetch", async () => {
    global.fetch = vi
      .fn()
      .mockResolvedValue({ ok: true, json: async () => ({}) });
    expect(await fetchHubFile("test", "file")).toBeDefined();
    expect(await parseModelIndex("test")).toBeDefined();
  });

  it("should set config", () => {
    setProgressBarConfig(false);
    expect(globalProgressBarConfig.enabled).toBe(false);
  });
});
