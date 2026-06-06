import { describe, expect, it, vi } from "vitest";
import { Toolbar } from "../src/components/toolbar.js";

describe("Toolbar", () => {
  it("should render and bind", () => {
    const container = document.createElement("div");
    const cfg: any = {
      onCleanGraph: vi.fn(),
      onMakeDynamic: vi.fn(),
      onToggleStrict: vi.fn(),
    };
    new Toolbar(container, cfg);

    const cleanBtn = container.querySelector("button") as HTMLButtonElement;
    cleanBtn.click();
    expect(cfg.onCleanGraph).toHaveBeenCalled();
  });
});
