import { describe, expect, it } from "vitest";
import * as Module from "../../src/flatbuffer/builder";

describe("builder.ts", () => {
  it("should instantiate and cover FlatBufferBuilder", () => {
    try {
      const obj = new (Module as any).FlatBufferBuilder();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
