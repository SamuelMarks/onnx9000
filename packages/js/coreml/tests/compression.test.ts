import { describe, it, expect } from "vitest";
import { applyCompression } from "../src/mil/compression.js";
import { TensorType } from "../src/mil/types.js";

describe("compression", () => {
  it("should compress", () => {
    const block: any = {
      operations: [
        {
          opType: "const",
          outputs: [{ type: new TensorType(1, [2, 2]) }],
          attributes: {},
        },
      ],
    };
    const res = applyCompression(block, {
      mode: "w8a16",
      reportReduction: true,
    });
    expect(res).toBeDefined();
    expect(res.compressedMemoryBytes).toBeDefined();
  });
});
