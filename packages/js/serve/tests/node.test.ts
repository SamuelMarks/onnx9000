import { describe, it, expect, vi } from "vitest";
import { serveNode } from "../src/node.js";

vi.mock("node:http", () => ({
  createServer: vi.fn().mockReturnValue({ listen: vi.fn() }),
}));

describe("node server", () => {
  it("should serve node", () => {
    const srv = serveNode({} as any, 8080);
    expect(srv).toBeDefined();
  });
});
