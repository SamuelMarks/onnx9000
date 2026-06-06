import { describe, it, expect } from "vitest";
import {
  DDIMScheduler,
  DDPMScheduler,
  EulerDiscreteScheduler,
  LCMScheduler,
} from "../src/schedulers.js";

describe("diffusers schedulers", () => {
  it("should step", () => {
    const ddim = new DDIMScheduler(100);
    const arr = [1, 2];
    expect(ddim.step([0, 0], 10, arr).length).toBe(2);

    const ddpm = new DDPMScheduler(100);
    expect(ddpm.step([0, 0], 10, arr).length).toBe(2);

    const euler = new EulerDiscreteScheduler(100);
    expect(euler.step([0, 0], 10, arr).length).toBe(2);

    const lcm = new LCMScheduler(100);
    expect(lcm.step([0, 0], 10, arr).length).toBe(2);
  });
});
