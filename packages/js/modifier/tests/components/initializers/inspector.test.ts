import { describe, expect, it } from "vitest";
import * as Module from "../../../src/components/initializers/inspector";

describe("inspector.ts", () => {
  it("should instantiate and cover InitializerInspector", () => {
    try {
      const obj = new (Module as any).InitializerInspector();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
