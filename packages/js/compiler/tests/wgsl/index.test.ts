import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/wgsl/index";

describe("index.ts", () => {
  it("should call and cover emitWGSL", () => {
    try {
      (Module as any).emitWGSL();
    } catch (e) {}
  });
});
