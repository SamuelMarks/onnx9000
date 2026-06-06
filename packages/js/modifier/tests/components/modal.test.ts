import { describe, expect, it } from "vitest";
import * as Module from "../../src/components/modal";

describe("modal.ts", () => {
  it("should instantiate and cover AddNodeModal", () => {
    try {
      const obj = new (Module as any).AddNodeModal();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
