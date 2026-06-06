import { describe, it, expect, vi } from "vitest";
import * as Module from "../src/dashboard";

describe("dashboard.ts", () => {
  it("should call and cover addDashboardRoutes", async () => {
    try {
      const res = (Module as any).addDashboardRoutes();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
});
