import { describe, expect, it } from "vitest";
import * as Module from "../src/interfaces";

describe("interfaces.ts", () => {
  it("should load module", () => {
    expect(Module).toBeDefined();
  });
});
