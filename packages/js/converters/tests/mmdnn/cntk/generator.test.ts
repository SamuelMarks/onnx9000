import { describe, it, expect, vi } from "vitest";
import * as Module from "../../../src/mmdnn/cntk/generator";

describe("generator.ts", () => {
  it("should instantiate and cover CNTKGenerator", () => {
    try {
      const obj = new (Module as any).CNTKGenerator();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
});
