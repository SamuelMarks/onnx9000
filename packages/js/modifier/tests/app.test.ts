import { Graph } from "@onnx9000/core";
import { describe, expect, it, vi } from "vitest";
import { __triggerCleanGraph, ModifierApp } from "../src/app.js";

vi.mock("../src/render/canvas.js", () => ({
  GraphRenderer: class {
    selectedNodeIds = new Set();
    render() {}
  },
}));

describe("ModifierApp", () => {
  it("should initialize", () => {
    const container = document.createElement("div");
    const g = new Graph("test");
    const app = new ModifierApp({ container, initialGraph: g });

    expect(app.editor).toBeDefined();

    __triggerCleanGraph(app);
  });
});
