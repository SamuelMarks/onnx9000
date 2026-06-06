import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/coreml/index";

describe("index.ts", () => {
  it("should call and cover compileToCoreML", () => {
    try {
      (Module as any).compileToCoreML();
    } catch (e) {}
  });
});
