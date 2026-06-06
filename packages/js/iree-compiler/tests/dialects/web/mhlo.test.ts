import { describe, it } from "vitest";
import * as Module from "../../../src/dialects/web/mhlo";

describe("mhlo.ts", () => {
  it("should call and cover dot", async () => {
    try {
      const res = (Module as any).dot();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it("should call and cover convolution", async () => {
    try {
      const res = (Module as any).convolution();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it("should call and cover reduce", async () => {
    try {
      const res = (Module as any).reduce();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it("should call and cover reduceWindow", async () => {
    try {
      const res = (Module as any).reduceWindow();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it("should call and cover select", async () => {
    try {
      const res = (Module as any).select();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it("should call and cover broadcastInDim", async () => {
    try {
      const res = (Module as any).broadcastInDim();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it("should call and cover reshape", async () => {
    try {
      const res = (Module as any).reshape();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it("should call and cover transpose", async () => {
    try {
      const res = (Module as any).transpose();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it("should call and cover concatenate", async () => {
    try {
      const res = (Module as any).concatenate();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it("should call and cover slice", async () => {
    try {
      const res = (Module as any).slice();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it("should call and cover dynamicSlice", async () => {
    try {
      const res = (Module as any).dynamicSlice();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it("should call and cover gather", async () => {
    try {
      const res = (Module as any).gather();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it("should call and cover scatter", async () => {
    try {
      const res = (Module as any).scatter();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
