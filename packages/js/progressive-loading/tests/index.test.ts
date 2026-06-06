import { describe, expect, it } from "vitest";
import { ProgressiveLoading } from "../src/index.js";

describe("ProgressiveLoading", () => {
  it("should run", () => {
    expect(new ProgressiveLoading().process("test")).toContain(
      "Progressive Loading",
    );
  });
});
