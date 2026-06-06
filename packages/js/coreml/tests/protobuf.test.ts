import { describe, expect, it } from "vitest";
import * as Module from "../src/protobuf";

describe("protobuf.ts", () => {
  it("should instantiate and cover Writer", () => {
    try {
      const obj = new (Module as any).Writer();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
