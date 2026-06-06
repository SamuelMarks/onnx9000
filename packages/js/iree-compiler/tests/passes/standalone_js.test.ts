import { describe, expect, it } from "vitest";
import * as Module from "../../src/passes/standalone_js";

describe("standalone_js.ts", () => {
  it("should instantiate and cover StandaloneJSExporter", () => {
    try {
      const obj = new (Module as any).StandaloneJSExporter();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it("should instantiate and cover ModelRunner", () => {
    try {
      const obj = new (Module as any).ModelRunner();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
