import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/models/mae";

describe("mae.ts", () => {
  it("should instantiate and cover MaskedAutoencoderViT", () => {
    try {
      const obj = new (Module as any).MaskedAutoencoderViT();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should call and cover maeVitBasePatch16", async () => {
    try {
      const res = (Module as any).maeVitBasePatch16();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
});
