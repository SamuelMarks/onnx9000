import { describe, expect, it } from "vitest";
import { GGUFTensorType, GGUFWriter } from "../src/builder.js";
import { GGUFReader } from "../src/reader.js";
import {
  reconstructONNX,
  reverseMapName,
  reverseMapType,
} from "../src/reverse.js";

describe("reverse", () => {
  it("should reverse map name", () => {
    expect(reverseMapName("token_embd.weight")).toBe(
      "model.embed_tokens.weight",
    );
    expect(reverseMapType(GGUFTensorType.F32)).toBe("float32");
  });

  it("should reconstruct", () => {
    const w = new GGUFWriter();
    w.addString("general.architecture", "llama");
    w.addTensorInfo("t", [1n], GGUFTensorType.F32, 0n);
    const buf = new ArrayBuffer(w.getHeaderSize() + 32);
    w.writeHeader(buf);

    const r = new GGUFReader(buf);
    const g = reconstructONNX(r);
    expect(g).toBeDefined();
  });
});
