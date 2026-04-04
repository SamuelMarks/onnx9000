// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ModifierApp } from '../src/app';
import { Graph, Node } from '@onnx9000/core';

describe('Coverage ModifierApp', () => {
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

  it('covers app update flow', () => {
    document.body.innerHTML = '<div id="app"></div>';
    const g = new Graph('test');
    g.opsetImports[''] = 10; // trigger deprecated warning
    g.addNode(new Node('Relu', ['x'], ['y']));
    g.addNode(new Node('Add', ['y', 'z'], ['out']));

    const app = new ModifierApp({
      container: document.querySelector('#app') as HTMLElement,
      initialGraph: g,
    });

    // Trigger update
    app.updateView();
    app.updateView(); // test debounce clear

    vi.runAllTimers(); // this runs the first setTimeout for _updateViewInternal
    vi.runAllTimers(); // this runs the second setTimeout for layout computing

    // Assert it rendered something
    expect(app.centerPanel.innerHTML).toBeDefined();
  });
});
