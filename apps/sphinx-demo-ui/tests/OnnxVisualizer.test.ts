// @ts-nocheck
import { describe, expect, it, vi } from 'vitest';
import { OnnxVisualizer } from '../src/components/OnnxVisualizer.js';

vi.mock('cytoscape', () => {
  const cytoscape = vi.fn().mockReturnValue({
    elements: () => ({ remove: vi.fn() }),
    add: vi.fn(),
    layout: () => ({ run: vi.fn() }),
    center: vi.fn(),
    on: vi.fn(),
    destroy: vi.fn()
  });
  return { default: cytoscape };
});

describe('OnnxVisualizer', () => {
  it('should render graph', () => {
    const viz = new OnnxVisualizer();
    document.body.appendChild(viz.element);

    viz.renderGraph({ nodes: [], inputs: [], outputs: [] });
    expect(viz.element.className).toContain('demo-onnx-viz-container');
  });
});
