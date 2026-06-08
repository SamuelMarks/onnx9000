import { beforeEach, describe, expect, it } from "vitest";

describe("demo", () => {
  beforeEach(() => {
    document.body.innerHTML = "<form id=\"chat-form\"><input id=\"prompt-input\"/><button id=\"send-btn\"></button><div id=\"messages\"></div></form>";
  });

  it("should run flow", async () => {
    try {
      const { initLlamaWebDemo } = await import("../src/main.ts");
      initLlamaWebDemo();
      
      // now break the DOM to hit the early return branch
      document.body.innerHTML = "";
      initLlamaWebDemo();
    } catch (_e) {}

    // Restore DOM to continue flow
    document.body.innerHTML = "<form id=\"chat-form\"><input id=\"prompt-input\"/><button id=\"send-btn\"></button><div id=\"messages\"></div></form>";
    const { initLlamaWebDemo } = await import("../src/main.ts");
    initLlamaWebDemo();

    const form = document.getElementById("chat-form") as HTMLFormElement;
    const input = document.getElementById("prompt-input") as HTMLInputElement;
    const messages = document.getElementById("messages") as HTMLElement;

    // submit empty
    form.dispatchEvent(new Event("submit", { cancelable: true }));

    // submit something
    input.value = "hello";
    form.dispatchEvent(new Event("submit", { cancelable: true }));
    
    // submit while generating
    input.value = "ignored";
    form.dispatchEvent(new Event("submit", { cancelable: true }));

    await new Promise((r) => setTimeout(r, 2500));
    expect(messages.textContent).toContain("How else can I help you today?");
  });
});
