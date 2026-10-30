import { Graph } from '@onnx9000/core';
import { describe, expect, it, vi } from 'vitest';
import { GraphRenderer } from '../src/render/canvas.js';

describe('GraphRenderer', () => {
  it('should render graph', () => {
    const canvas = document.createElement('canvas');
    canvas.getContext = vi.fn().mockReturnValue({
      clearRect: vi.fn(),
      fillRect: vi.fn(),
      strokeRect: vi.fn(),
      setTransform: vi.fn(),
      save: vi.fn(),
      restore: vi.fn(),
      translate: vi.fn(),
      scale: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      arc: vi.fn(),
      bezierCurveTo: vi.fn(),
      stroke: vi.fn(),
      fill: vi.fn(),
      fillText: vi.fn(),
      roundRect: vi.fn(),
      measureText: vi.fn().mockReturnValue({ width: 10 }),
    });
    const renderer = new GraphRenderer(canvas);
    const g = new Graph('test');

    renderer.render(g, {
      nodes: new Map(),
      edges: [],
      bounds: { width: 100, height: 100 },
    });
    expect(renderer).toBeDefined();
    renderer.destroy();
  });
});
