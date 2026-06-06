import { describe, expect, it } from "vitest";
import * as Module from "../src/enums";

describe("enums.ts", () => {
  it("should load module", () => {
    expect(Module).toBeDefined();
  });
});
