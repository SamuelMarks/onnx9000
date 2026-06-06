import { describe, expect, it, vi } from "vitest";
import { runCli } from "../src/cli.js";

vi.mock("../src/index.js", () => ({
  createServer: vi.fn().mockReturnValue({}),
  serveNode: vi.fn(),
}));

describe("serve cli", () => {
  it("should run", () => {
    expect(() => runCli(["--port", "8080"])).not.toThrow();
  });
});
