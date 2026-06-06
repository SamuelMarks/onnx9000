import * as fs from "node:fs";
import { describe, expect, it, vi } from "vitest";
import { main } from "../bin/cli.js";

vi.mock("fs", () => ({
  default: {
    readFileSync: vi.fn().mockReturnValue(new Uint8Array()),
    writeFileSync: vi.fn(),
    existsSync: vi.fn().mockReturnValue(true),
    mkdirSync: vi.fn(),
  },
}));

vi.mock("@onnx9000/core", () => ({
  load: vi.fn().mockReturnValue({
    inputs: [],
    outputs: [],
    nodes: [],
    valueInfo: [],
    initializers: [],
    tensors: {},
  }),
}));

vi.mock("../dist/index.js", () => ({
  OpenVinoExporter: class {
    export() {
      return { xml: "x", bin: new Uint8Array() };
    }
  },
}));

describe("openvino cli", () => {
  it("should run main", () => {
    const origArgs = process.argv;
    process.argv = ["node", "cli", "test.onnx"];

    const mockLog = vi.spyOn(console, "log").mockImplementation(() => {});
    const mockExit = vi
      .spyOn(process, "exit")
      .mockImplementation((() => {}) as any);

    main();

    expect(fs.default.writeFileSync).toHaveBeenCalled();

    process.argv = origArgs;
    mockLog.mockRestore();
    mockExit.mockRestore();
  });
});
