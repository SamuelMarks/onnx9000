import { describe, it, expect } from "vitest";
import { GraphEditor } from "../src/ui/editor.js";
import { Graph } from "@onnx9000/core";

describe("GraphEditor", () => {
  it("should select and delete", () => {
    const g = new Graph("test");
    const mutator: any = { removeNode: () => {} };
    const ed = new GraphEditor(g, mutator);

    ed.selectNode("1");
    expect(ed.selectedNodeIds.has("1")).toBe(true);

    ed.deleteSelection();
    expect(ed.selectedNodeIds.has("1")).toBe(false);
  });
});
