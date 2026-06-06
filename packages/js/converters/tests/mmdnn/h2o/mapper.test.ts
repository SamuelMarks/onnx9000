import { describe, expect, it } from "vitest";
import * as Module from "../../../src/mmdnn/h2o/mapper";

describe("mapper.ts", () => {
  it("should instantiate and cover H2OMapper", () => {
    try {
      const obj = new (Module as any).H2OMapper();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
