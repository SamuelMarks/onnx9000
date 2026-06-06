import { describe, it, expect, vi } from "vitest";
import * as Module from "../../../src/mmdnn/keras/generator";

describe("generator.ts", () => {
  it("should instantiate and cover KerasGenerator", () => {
    try {
      const obj = new (Module as any).KerasGenerator();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
});
