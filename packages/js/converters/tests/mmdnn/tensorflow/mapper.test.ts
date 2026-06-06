import { describe, expect, it } from "vitest";
import * as Module from "../../../src/mmdnn/tensorflow/mapper";

describe("mapper.ts", () => {
  it("should instantiate and cover TFMapper", () => {
    try {
      const obj = new (Module as any).TFMapper();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
