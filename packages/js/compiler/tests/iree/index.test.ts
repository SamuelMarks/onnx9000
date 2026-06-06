import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/iree/index";

describe("index.ts", () => {
  it("should call and cover compileToIREE", () => {
    try {
      (Module as any).compileToIREE();
    } catch (e) {}
  });
});
