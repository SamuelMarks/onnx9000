import { describe, it, expect } from "vitest";
import { mobilevitS } from "../src/models/mobilevit.js";

describe("MobileViT", () => {
  it("should create and call", () => {
    const model = mobilevitS();
    expect(model).toBeDefined();
    const out = model.call({} as any);
    expect(out).toBeDefined();
  });
});
