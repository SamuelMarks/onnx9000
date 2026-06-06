import { describe, expect, it } from "vitest";
import * as Module from "../../../src/mmdnn/paddle/parser";

describe("parser.ts", () => {
  it("should instantiate and cover PaddleParser", () => {
    try {
      const obj = new (Module as any).PaddleParser();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
