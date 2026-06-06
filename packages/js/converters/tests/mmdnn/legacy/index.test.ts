import { describe, expect, it } from "vitest";
import * as Module from "../../../src/mmdnn/legacy/index";

describe("index.ts", () => {
  it("should load module", () => {
    expect(Module).toBeDefined();
  });
});
