import { describe, it, expect } from "vitest";
import { WebNNLayoutValidator } from "../src/providers/webnn/conformance_validator.js";

describe("WebNNLayoutValidator", () => {
  it("should validate layouts", () => {
    const graph: any = {
      nodes: [
        {
          opType: "Conv",
          name: "c1",
          attributes: { layout: { value: "nchw" } },
        },
      ],
    };
    expect(WebNNLayoutValidator.validateLayouts(graph)).toBe(true);

    graph.nodes[0].attributes.layout.value = "invalid";
    expect(() => WebNNLayoutValidator.validateLayouts(graph)).toThrow();
  });
});
