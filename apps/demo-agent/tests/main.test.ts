import { beforeEach, describe, expect, it } from "vitest";

describe("demo", () => {
  beforeEach(() => {
    document.body.innerHTML =
      '<textarea id="prompt"></textarea><button id="runBtn"></button><div id="output"></div>';
  });

  it("should run flow", async () => {
    try {
      await import("../src/main.ts");
    } catch (_e) {}

    const btn = document.getElementById("runBtn") as HTMLButtonElement;
    const prompt = document.getElementById("prompt") as HTMLTextAreaElement;
    const out = document.getElementById("output") as HTMLElement;

    btn.click();

    prompt.value = "test";
    btn.click();

    await new Promise((r) => setTimeout(r, 4000));
    expect(out.innerText).toContain("[Agent] Final Answer: 55");
  });
});
