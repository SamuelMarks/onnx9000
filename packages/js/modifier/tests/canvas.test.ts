import { Graph } from "@onnx9000/core";
import { describe, expect, it } from "vitest";
import { GraphRenderer } from "../src/render/canvas.js";

describe("GraphRenderer", () => {
  it("should render graph", () => {
    const canvas = document.createElement("canvas");
    const renderer = new GraphRenderer(canvas);
    const g = new Graph("test");

    renderer.render(g, {
      nodes: new Map(),
      edges: [],
      bounds: { width: 100, height: 100 },
    });
    expect(renderer).toBeDefined();
    renderer.destroy();
  });
});
