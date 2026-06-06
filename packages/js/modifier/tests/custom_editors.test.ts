import { describe, it, expect } from "vitest";
import { renderCustomEditor } from "../src/components/editors/custom_editors.js";

describe("custom_editors", () => {
  it("should render conv editor", () => {
    const container = document.createElement("div");
    const node: any = { opType: "Conv", attributes: {} };
    const mutator: any = {};

    expect(renderCustomEditor({ container, node, mutator })).toBe(true);
    expect(container.innerHTML).toContain("Conv Settings");
  });

  it("should not render unknown", () => {
    const container = document.createElement("div");
    const node: any = { opType: "Unknown" };
    const mutator: any = {};

    expect(renderCustomEditor({ container, node, mutator })).toBe(false);
  });
});
