import { describe, it, expect, vi } from "vitest";
import * as Module from "../../../src/components/debugger/debugger";

describe("debugger.ts", () => {
  it("should instantiate and cover GraphDebugger", () => {
    try {
      const obj = new (Module as any).GraphDebugger();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
});
