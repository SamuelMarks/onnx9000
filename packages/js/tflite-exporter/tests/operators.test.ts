import { describe, expect, it } from "vitest";
import { mapOnnxNodeToTFLite } from "../src/compiler/operators.js";

describe("operators", () => {
  it("should map", () => {
    expect(
      mapOnnxNodeToTFLite({ opType: "Add", attributes: {} } as any),
    ).toBeDefined();
    expect(mapOnnxNodeToTFLite({ opType: "Unknown" } as any)).toBeNull();
  });
});
