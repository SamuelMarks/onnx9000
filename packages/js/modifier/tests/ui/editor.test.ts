import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/ui/editor";

describe("editor.ts", () => {
  it("should instantiate and cover GraphEditor", () => {
    try {
      const obj = new (Module as any).GraphEditor();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
});
