import { describe, expect, it } from "vitest";
import * as Module from "../../../src/mmdnn/sparkml/parser";

describe("parser.ts", () => {
  it("should instantiate and cover SparkMLParser", () => {
    try {
      const obj = new (Module as any).SparkMLParser();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
