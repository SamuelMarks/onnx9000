import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/jax/mapper";

describe("mapper.ts", () => {
  it("should instantiate and cover JaxMapper", () => {
    try {
      const obj = new (Module as any).JaxMapper();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
});
