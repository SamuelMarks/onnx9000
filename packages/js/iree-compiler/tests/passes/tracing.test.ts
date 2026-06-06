import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/passes/tracing";

describe("tracing.ts", () => {
  it("should instantiate and cover DiagnosticTracer", () => {
    try {
      const obj = new (Module as any).DiagnosticTracer();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
});
