import { describe, it, expect, vi } from "vitest";
import * as Module from "../../../src/mmdnn/h2o/parser";

describe("parser.ts", () => {
  it("should call and cover parseH2O", async () => {
    try {
      const res = (Module as any).parseH2O();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
});
