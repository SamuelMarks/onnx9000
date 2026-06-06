import { describe, it } from "vitest";
import { inferShapes } from "../src/mil/rewriter.js";

describe("rewriter", () => {
  it("should infer shapes", () => {
    const block: any = {
      inputs: [],
      outputs: [],
      operations: [{ opType: "const", outputs: [{ type: { shape: [1] } }] }],
    };
    inferShapes(block);
  });
});
