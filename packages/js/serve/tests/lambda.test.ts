import { describe, expect, it, vi } from "vitest";
import { createLambdaHandler } from "../src/lambda.js";

describe("lambda", () => {
  it("should handle request", async () => {
    const srv = {
      fetch: vi.fn().mockResolvedValue(new Response("ok", { status: 200 })),
    };
    const handler = createLambdaHandler(srv as any);

    const event = {
      httpMethod: "GET",
      path: "/",
      headers: { host: "test" },
    };
    const ctx = {
      getRemainingTimeInMillis: () => 1000,
    };

    const res = await handler(event, ctx);
    expect(res.statusCode).toBe(200);
    expect(res.body).toBe("ok");
  });
});
