import { describe, expect, it } from "vitest";
import * as t from "../src/index.js";

describe("Index", () => {
  it("exports everything", () => {
    expect(t).toBeDefined();
  });
});
