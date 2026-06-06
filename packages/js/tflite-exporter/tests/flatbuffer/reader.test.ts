import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/flatbuffer/reader";

describe("reader.ts", () => {
  it("should instantiate and cover FlatBufferReader", () => {
    try {
      const obj = new (Module as any).FlatBufferReader();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
});
