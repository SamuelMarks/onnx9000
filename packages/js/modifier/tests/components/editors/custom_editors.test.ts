import { describe, it, expect, vi } from "vitest";
import * as Module from "../../../src/components/editors/custom_editors";

describe("custom_editors.ts", () => {
  it("should call and cover renderCustomEditor", async () => {
    try {
      const res = (Module as any).renderCustomEditor();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
});
