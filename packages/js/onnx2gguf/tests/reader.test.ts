import { describe, it, expect } from "vitest";
import { GGUFReader } from "../src/reader.js";
import { GGUFWriter, GGUFValueType, GGUFTensorType } from "../src/builder.js";

describe("GGUFReader", () => {
  it("should read", () => {
    const w = new GGUFWriter();
    w.addUint8("u8", 1);
    const size = w.getHeaderSize();
    const buf = new ArrayBuffer(size);
    w.writeHeader(buf);

    const r = new GGUFReader(buf);
    expect(r.kvs["u8"]).toBe(1);
  });
});
