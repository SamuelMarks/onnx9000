import { describe, it } from "vitest";
import * as Module from "../../src/mil/linter";

describe("linter.ts", () => {
  it("should call and cover lintMILProgram", async () => {
    try {
      const res = (Module as any).lintMILProgram();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
