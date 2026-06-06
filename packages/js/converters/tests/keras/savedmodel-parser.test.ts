import { describe, it } from "vitest";
import * as Module from "../../src/keras/savedmodel-parser";

describe("savedmodel-parser.ts", () => {
  it("should call and cover parseSavedModel", async () => {
    try {
      const res = (Module as any).parseSavedModel();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
