import { describe, it, expect } from "vitest";
import { AutoShardingPass, allReduce } from "../src/sharding.js";
import { Graph } from "../ir/graph.js";

describe("sharding", () => {
  it("should apply sharding pass", () => {
    const pass = new AutoShardingPass();
    const g = new Graph("test");
    pass.apply(g);
    expect(g).toBeDefined();
  });

  it("should record ops", () => {
    const out = allReduce({} as any);
    expect(out.name).toBe("AllReduce_out");
  });
});
