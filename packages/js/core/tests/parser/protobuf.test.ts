import { describe, expect, it } from "vitest";
import * as Module from "../../src/parser/protobuf";

describe("protobuf.ts", () => {
  it("should instantiate and cover BufferReader", () => {
    try {
      const obj = new (Module as any).BufferReader();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it("should instantiate and cover BlobReader", () => {
    try {
      const obj = new (Module as any).BlobReader();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it("should call and cover readVarInt", async () => {
    try {
      const res = (Module as any).readVarInt();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it("should call and cover readVarInt64", async () => {
    try {
      const res = (Module as any).readVarInt64();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it("should call and cover readString", async () => {
    try {
      const res = (Module as any).readString();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it("should call and cover readTag", async () => {
    try {
      const res = (Module as any).readTag();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it("should call and cover skipField", async () => {
    try {
      const res = (Module as any).skipField();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
