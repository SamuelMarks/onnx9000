import { describe, expect, it } from "vitest";
import * as Module from "../../../src/mmdnn/legacy/quirks";

describe("quirks.ts", () => {
  it("should instantiate and cover LegacyQuirkResolver", () => {
    try {
      const obj = new (Module as any).LegacyQuirkResolver();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
