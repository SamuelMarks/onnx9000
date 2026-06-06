// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from "vitest";
import { LayoutBuilder } from "../src/components/layout.js";

describe("LayoutBuilder", () => {
  let container: HTMLElement;
  let builder: LayoutBuilder;

  beforeEach(() => {
    container = document.createElement("div");
    builder = new LayoutBuilder(container);
  });

  it("builds a layout", () => {
    const result = builder.build();
    expect(result.leftPanel).toBeDefined();
    expect(result.centerPanel).toBeDefined();
    expect(result.rightPanel).toBeDefined();

    expect(result.leftPanel.id).toBe("modifier-left-panel");
    expect(result.centerPanel.id).toBe("modifier-center-panel");
    expect(result.rightPanel.id).toBe("modifier-right-panel");
  });
});
