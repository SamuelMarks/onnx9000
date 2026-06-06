import { describe, it, expect, vi } from "vitest";
import { validateMILProgram, validateBlock } from "../src/mil/validator.js";

vi.mock("../src/sort.js", () => ({ topologicalSort: vi.fn() }));

describe("validator", () => {
  it("should validate", () => {
    const block: any = {
      name: "test",
      inputs: [{ name: "in" }],
      outputs: [{ name: "out" }],
      operations: [
        {
          opType: "const",
          inputs: { x: { name: "in" } },
          outputs: [{ name: "out" }],
        },
      ],
    };
    expect(() => validateBlock(block)).not.toThrow();
  });
});
