import { describe, it, expect } from "vitest";
import { addOpenAIRoutes } from "../src/openai.js";
import { Router } from "../src/router.js";

describe("openai", () => {
  it("should handle chat", async () => {
    const r = new Router();
    addOpenAIRoutes({} as any, r);

    const req = new Request("http://localhost/v1/chat/completions", {
      method: "POST",
      body: JSON.stringify({ messages: [{ role: "user", content: "test" }] }),
    });

    const res = await r.handle(req);
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toContain("Hello from onnx9000-model");
  });
});
