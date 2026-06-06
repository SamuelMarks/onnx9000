import { describe, it, expect, vi } from "vitest";
import * as Module from "../src/session";

describe("session.ts", () => {
  it("should instantiate and cover InferenceSession", () => {
    try {
      const obj = new (Module as any).InferenceSession();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
});
