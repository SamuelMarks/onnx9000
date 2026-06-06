import { describe, it, expect } from "vitest";
import { emitWGSL } from "../src/wgsl/index.js";

describe("wgsl index", () => {
  it("should generate wgsl", () => {
    const graph: any = {
      inputs: [{ name: "in1" }, { name: "in2" }],
      outputs: ["out"],
      nodes: [{ opType: "Add", inputs: ["in1", "in2"], outputs: ["out"] }],
    };

    const code = emitWGSL(graph);
    expect(code).toContain("@compute");
    expect(code).toContain("in1[i] + in2[i]");
  });

  it("should throw on empty graph", () => {
    expect(() => emitWGSL({ nodes: [] } as any)).toThrow("Graph is empty");
  });
});
