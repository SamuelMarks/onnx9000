import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@onnx9000/core", () => {
  return {
    Graph: class {
      nodes: any[] = [];
      tensors: Record<string, any> = {};
      inputs: any[] = [];
      outputs: any[] = [];
      initializers: any[] = [];
    },
  };
});

vi.mock("@onnx9000/modifier/dist/GraphMutator.js", () => ({
  GraphMutator: class {},
}));

vi.mock("@onnx9000/modifier/dist/components/export/exporter.js", () => ({
  ModelExporter: class {},
}));

vi.mock("../src/render/canvas", () => ({
  CanvasRenderer: class {
    setLayout() {}
    setFilterControlEdges() {}
    setCustomColorRegex() {}
    setSearchResults() {}
    focusNode() {}
    render() {}
    selectedNodes = [];
  },
}));

describe("index.ts", () => {
  beforeEach(() => {
    document.body.innerHTML = "";
    vi.resetModules();
    (global as any).Worker = class Worker {
      postMessage() {}
      onmessage() {}
    };
  });

  it("should initialize UI on import", async () => {
    await import("../src/index.ts");
    expect(document.getElementById("view")).not.toBeNull();
    expect(document.getElementById("status")).not.toBeNull();
  });

  it("should handle file drop event on window", async () => {
    await import("../src/index.ts");
    const mockFile = new File(["mock content"], "model.onnx", {
      type: "application/octet-stream",
    });
    const dropEvent = new Event("drop") as any;
    dropEvent.dataTransfer = { files: [mockFile] };
    window.dispatchEvent(dropEvent);

    const dropZone = document.getElementById("drop-zone")!;
    expect(dropZone.textContent).toContain("Loaded: model.onnx");
  });
});
