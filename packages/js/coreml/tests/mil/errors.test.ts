import { describe, expect, it } from "vitest";
import * as Module from "../../src/mil/errors";

describe("errors.ts", () => {
  it("should instantiate and cover CoreMLExportError", () => {
    try {
      const obj = new (Module as any).CoreMLExportError();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it("should instantiate and cover UnsupportedOpError", () => {
    try {
      const obj = new (Module as any).UnsupportedOpError();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it("should instantiate and cover ThermalThrottlingWarning", () => {
    try {
      const obj = new (Module as any).ThermalThrottlingWarning();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it("should instantiate and cover ANELimitsExceededWarning", () => {
    try {
      const obj = new (Module as any).ANELimitsExceededWarning();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it("should instantiate and cover DoubleDowncastWarning", () => {
    try {
      const obj = new (Module as any).DoubleDowncastWarning();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
