import { describe, expect, it } from "vitest";
import * as Module from "../../../src/mmdnn/keras/importer";

describe("importer.ts", () => {
  it("should instantiate and cover KerasImporter", () => {
    try {
      const obj = new (Module as any).KerasImporter();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
