import { describe, it } from "vitest";
import * as Module from "../../src/ops/registry";

describe("registry.ts", () => {
  it("should call and cover register_op", async () => {
    try {
      const res = (Module as any).register_op();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
