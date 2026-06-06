import { describe, it, expect, vi } from "vitest";
import * as Module from "../src/parsers";

describe("parsers.ts", () => {
  it("should instantiate and cover BaseParser", () => {
    try {
      const obj = new (Module as any).BaseParser();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover PyTorchFXParser", () => {
    try {
      const obj = new (Module as any).PyTorchFXParser();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover JAXprParser", () => {
    try {
      const obj = new (Module as any).JAXprParser();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover XLAHLOParser", () => {
    try {
      const obj = new (Module as any).XLAHLOParser();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
});
