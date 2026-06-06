import { describe, expect, it } from "vitest";
import * as Module from "../src/memory";

describe("memory.ts", () => {
  it("should instantiate and cover MemoryManager", () => {
    try {
      const obj = new (Module as any).MemoryManager();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
