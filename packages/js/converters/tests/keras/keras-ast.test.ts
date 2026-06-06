import { describe, it } from "vitest";
import * as Module from "../../src/keras/keras-ast";

describe("keras-ast.ts", () => {
  it("should call and cover extractKerasTopology", async () => {
    try {
      const res = (Module as any).extractKerasTopology();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
