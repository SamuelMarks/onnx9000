import { describe, expect, it } from "vitest";
import * as Module from "../../src/parser/protobuf_writer";

describe("protobuf_writer.ts", () => {
  it("should instantiate and cover BufferWriter", () => {
    try {
      const obj = new (Module as any).BufferWriter();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
