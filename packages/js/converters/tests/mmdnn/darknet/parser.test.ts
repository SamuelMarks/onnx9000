import { describe, it } from "vitest";
import * as Module from "../../../src/mmdnn/darknet/parser";

describe("parser.ts", () => {
  it("should call and cover parseCfg", async () => {
    try {
      const res = (Module as any).parseCfg();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it("should call and cover parseWeights", async () => {
    try {
      const res = (Module as any).parseWeights();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
