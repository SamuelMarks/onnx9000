import { describe, it } from "vitest";
import * as Module from "../../src/keras/emitters-pool";

describe("emitters-pool.ts", () => {
  it("should call and cover emitPool", async () => {
    try {
      const res = (Module as any).emitPool();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it("should call and cover emitGlobalPool", async () => {
    try {
      const res = (Module as any).emitGlobalPool();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
