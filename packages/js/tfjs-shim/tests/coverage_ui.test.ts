import { describe, expect, it } from "vitest";
import { TfjsShimDemoElement } from "../src/ui.js";

describe("Coverage UI", () => {
  it("TfjsShimDemoElement", () => {
    const el = new TfjsShimDemoElement();

    const _html = "";
    let _clickListener: Object;
    const _mockShadow = {
      innerHTML: "",
      querySelector: (sel: string) => {
        if (sel === "#run-btn") {
          return {
            addEventListener: (_evt: string, cb: Object) => {
              _clickListener = cb;
            },
          };
        }
        if (sel === "#results") {
          return { textContent: "" };
        }
      },
    };

    // vitest with jsdom already provides HTMLElement and attachShadow, so let's try standard DOM API
    if (typeof document !== "undefined") {
      document.body.appendChild(el);
      const btn = el.shadowRoot?.querySelector("#run-btn") as HTMLButtonElement;
      btn.click();
      const results = el.shadowRoot?.querySelector("#results");
      expect(results?.textContent).toContain("Results match!");
    }
  });
});
