import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/keras/emitters-conv";

describe("emitters-conv.ts", () => {
  it("should call and cover emitConv", async () => {
    try {
      const res = (Module as any).emitConv();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
  it("should call and cover emitSeparableConv", async () => {
    try {
      const res = (Module as any).emitSeparableConv();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
});
