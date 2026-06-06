import { describe, expect, it } from "vitest";
import { Mlirlowering } from "../src/index.js";

describe("Mlirlowering", () => {
  it("should run", () => {
    expect(new Mlirlowering().run()).toBeDefined();
  });
});
