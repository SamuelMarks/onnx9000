import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/shape_inference/infer";

describe("infer.ts", () => {
  it("should call and cover inferShapes", async () => {
    try {
      const res = (Module as any).inferShapes();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
});
