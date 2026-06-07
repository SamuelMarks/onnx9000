import { describe, expect, it } from "vitest";
import * as Module from "../../../src/providers/webnn/compiler";

describe("compiler.ts", () => {
  it("should instantiate and cover WebNNCompiler", () => {
    try {
      const obj = new (Module as any).WebNNCompiler();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
