import { describe, it, expect } from "vitest";
import { convnextTiny } from "../src/models/convnext.js";

describe("ConvNeXt", () => {
  it("should create and call", () => {
    const model = convnextTiny();
    expect(model).toBeDefined();
    const out = model.call({} as any);
    expect(out).toBeDefined();
  });
});
