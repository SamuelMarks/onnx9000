import { beforeEach, describe, expect, it } from "vitest";

describe("demo", () => {
  beforeEach(() => {
    document.body.innerHTML = "<button id=\"convert-btn\"></button><div id=\"output\"></div>";
  });

  it("should run flow", async () => {
    try {
      const { initTensorrtDemo } = await import("../src/main.ts");
      // break dom
      document.body.innerHTML = "";
      initTensorrtDemo();
    } catch (_e) {}

    document.body.innerHTML = "<button id=\"convert-btn\"></button><div id=\"output\"></div>";
    const { initTensorrtDemo } = await import("../src/main.ts");
    initTensorrtDemo();

    const btn = document.getElementById("convert-btn") as HTMLButtonElement;
    const out = document.getElementById("output") as HTMLElement;

    btn.click();
    expect(out.innerText).toContain("import tensorrt as trt");
  });
});
