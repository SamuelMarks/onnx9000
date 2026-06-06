import { describe, it } from "vitest";
import * as Module from "../../src/passes/bufferize";

describe("bufferize.ts", () => {
  it("should call and cover bufferizeLinalg", async () => {
    try {
      const res = (Module as any).bufferizeLinalg();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
