import { describe, it, expect } from "vitest";
import { Router } from "../src/router.js";

describe("Router", () => {
  it("should route", async () => {
    const r = new Router();
    r.get("/test/:id", async (req, params) => new Response(params.id));

    const res = await r.handle(new Request("http://localhost/test/123"));
    expect(await res.text()).toBe("123");
  });

  it("should handle options", async () => {
    const r = new Router();
    const res = await r.handle(
      new Request("http://localhost/test", { method: "OPTIONS" }),
    );
    expect(res.status).toBe(204);
  });
});
