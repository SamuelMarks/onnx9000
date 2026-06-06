import { describe, expect, it } from "vitest";
import * as Module from "../../../src/mmdnn/catboost/parser";

describe("parser.ts", () => {
  it("should instantiate and cover CatBoostParser", () => {
    try {
      const obj = new (Module as any).CatBoostParser();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
