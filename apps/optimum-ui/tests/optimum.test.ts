import { describe, expect, it } from "vitest";
import * as opt from "../src/index.js";

describe("optimum-ui", () => {
  it("should export nothing or something", () => {
    expect(opt).toBeDefined();
  });
});
