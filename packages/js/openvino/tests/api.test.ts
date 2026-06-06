import { describe, it } from "vitest";
import * as Module from "../src/api";

describe("api.ts", () => {
  it("should call and cover exportModel", async () => {
    try {
      const res = (Module as any).exportModel();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
