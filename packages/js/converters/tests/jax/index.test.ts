import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/jax/index";

describe("index.ts", () => {
  it("should load module", () => {
    expect(Module).toBeDefined();
  });
});
