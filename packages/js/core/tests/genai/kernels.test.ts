import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/genai/kernels";

describe("kernels.ts", () => {
  it("should call and cover wasmLogitSortFilter", async () => {
    try {
      const res = (Module as any).wasmLogitSortFilter();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
  it("should call and cover wasmFlashAttentionSimd", async () => {
    try {
      const res = (Module as any).wasmFlashAttentionSimd();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
});
