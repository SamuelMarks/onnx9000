// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ModifierApp } from '../src/app';
import { Graph, Node } from '@onnx9000/core';

describe('Coverage Modifier App interactions', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    const handler = {
      get(target: any, prop: string) {
        if (prop === 'measureText') return vi.fn().mockReturnValue({ width: 10 });
        if (prop in target) return target[prop];
        return vi.fn();
      },
    };
    HTMLCanvasElement.prototype.getContext = () => new Proxy({}, handler);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('covers autoFix and duplicated nodes via keyboard', () => {
    document.body.innerHTML = '<div id="app"></div>';
    const g = new Graph('test');
    g.addNode(new Node('Add', ['a', 'b'], ['c']));
    const app = new ModifierApp({
      container: document.querySelector('#app') as HTMLElement,
      initialGraph: g,
    });

    // 1. autoFixMissingInitializers
    if ((app as any).initializerInspector && (app as any).initializerInspector.config) {
      if ((app as any).initializerInspector.config.onAutoFix)
        (app as any).initializerInspector.config.onAutoFix();
      if ((app as any).initializerInspector.config.onStripInitializers)
        (app as any).initializerInspector.config.onStripInitializers();
    }

    // 2. Keyboard event for duplicate: ctrl+d
    app.editor.selectedNodeIds.add(g.nodes[0].id);
    const event = new window.KeyboardEvent('keydown', { key: 'd', ctrlKey: true });
    window.dispatchEvent(event);

    vi.runAllTimers();
  });
});
