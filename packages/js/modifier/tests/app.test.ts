import { describe, it, expect, vi } from 'vitest';
import { ModifierApp, __triggerCleanGraph } from '../src/app.js';
import { Graph } from '@onnx9000/core';

vi.mock('../src/render/canvas.js', () => ({
  GraphRenderer: class {
    selectedNodeIds = new Set();
    render() {}
  },
}));

describe('ModifierApp', () => {
  it('should initialize', () => {
    const container = document.createElement('div');
    const g = new Graph('test');
    const app = new ModifierApp({ container, initialGraph: g });

    expect(app.editor).toBeDefined();

    __triggerCleanGraph(app);
  });
});
