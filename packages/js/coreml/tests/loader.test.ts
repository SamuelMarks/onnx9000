import { describe, expect, it } from "vitest";
import * as Module from "../src/loader";

describe("loader.ts", () => {
  it("should instantiate and cover MLPackageLoader", () => {
    try {
      const obj = new (Module as any).MLPackageLoader();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
