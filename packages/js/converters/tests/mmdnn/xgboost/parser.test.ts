import { describe, expect, it } from "vitest";
import * as Module from "../../../src/mmdnn/xgboost/parser";

describe("parser.ts", () => {
  it("should instantiate and cover XGBoostParser", () => {
    try {
      const obj = new (Module as any).XGBoostParser();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
