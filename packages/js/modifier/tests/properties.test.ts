import { describe, expect, it } from "vitest";
import { PropertiesPanel } from "../src/components/properties.js";

describe("PropertiesPanel", () => {
  it("should render graph properties", () => {
    const container = document.createElement("div");
    const mutator: any = {};
    const panel = new PropertiesPanel(container, mutator);

    const g: any = {
      name: "test",
      inputs: [],
      outputs: [],
      valueInfo: [],
      nodes: [],
      initializers: [],
      tensors: {},
      opsetImports: {},
    };
    panel.renderGraphProperties(g);

    expect((container.querySelector("input") as HTMLInputElement).value).toBe(
      "test",
    );
  });
});
