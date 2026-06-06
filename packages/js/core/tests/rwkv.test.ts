import { describe, it, expect } from "vitest";
import { rwkvV4 } from "../src/models/rwkv.js";

describe("RWKV", () => {
  it("should create and call", () => {
    const model = rwkvV4();
    expect(model).toBeDefined();
    const out = model.call({} as any);
    expect(out).toBeDefined();
  });
});
