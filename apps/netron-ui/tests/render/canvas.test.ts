import { describe, it, expect, vi, beforeEach } from "vitest";
import { CanvasRenderer } from "../../src/render/canvas.ts";

// polyfill Path2D for JSDOM
(global as any).Path2D = class Path2D {
  moveTo() {}
  bezierCurveTo() {}
  lineTo() {}
};

describe("CanvasRenderer", () => {
  let canvas: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D;

  beforeEach(() => {
    canvas = document.createElement("canvas");
    ctx = {
      clearRect: vi.fn(),
      save: vi.fn(),
      translate: vi.fn(),
      scale: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      setLineDash: vi.fn(),
      restore: vi.fn(),
      roundRect: vi.fn(),
      fill: vi.fn(),
      fillText: vi.fn(),
      measureText: vi.fn().mockReturnValue({ width: 10 }),
      bezierCurveTo: vi.fn(),
      isPointInStroke: vi.fn().mockReturnValue(false),
    } as unknown as CanvasRenderingContext2D;
    vi.spyOn(canvas, "getContext").mockReturnValue(ctx);
  });

  it("should initialize correctly", () => {
    const renderer = new CanvasRenderer(canvas);
    expect(renderer).toBeDefined();
  });
});
