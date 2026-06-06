import * as fs from "node:fs";
import { describe, expect, it, vi } from "vitest";
import { main } from "../src/cli/index.js";

vi.mock("fs", () => ({
  existsSync: vi.fn().mockReturnValue(true),
  writeFileSync: vi.fn(),
}));

describe("cli index", () => {
  it("should parse args and run", () => {
    const origArgs = process.argv;
    process.argv = ["node", "cli", "compile", "test.onnx"];

    const mockExit = vi
      .spyOn(process, "exit")
      .mockImplementation((() => {}) as any);
    const mockLog = vi.spyOn(console, "log").mockImplementation(() => {});

    main();

    expect(fs.writeFileSync).toHaveBeenCalled();
    expect(mockLog).toHaveBeenCalledWith(
      expect.stringContaining("Successfully generated"),
    );

    process.argv = origArgs;
    mockExit.mockRestore();
    mockLog.mockRestore();
  });
});
