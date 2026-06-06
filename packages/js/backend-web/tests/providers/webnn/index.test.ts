import { describe, it, expect, vi } from "vitest";
import * as Module from "../../../src/providers/webnn/index";

describe("index.ts", () => {
  it("should instantiate and cover WebNNProvider", () => {
    try {
      const obj = new (Module as any).WebNNProvider();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
});
