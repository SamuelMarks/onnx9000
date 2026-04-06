// @vitest-environment jsdom
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { Graph, Node } from '@onnx9000/core';
import { ModifierApp } from '../src/app.js';

describe('ModifierApp (Integration)', () => {
  let container: HTMLElement;
  let graph: Graph;

  beforeEach(() => {
    container = document.createElement('div');
    document.body.appendChild(container);
    const mockCtx = {
      setTransform: vi.fn(),
      save: vi.fn(),
      restore: vi.fn(),
      fillStyle: '',
      strokeStyle: '',
      lineWidth: 1,
      fillRect: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      quadraticCurveTo: vi.fn(),
      closePath: vi.fn(),
      fill: vi.fn(),
      font: '',
      textAlign: '',
      textBaseline: '',
      fillText: vi.fn(),
      setLineDash: vi.fn(),
      strokeRect: vi.fn(),
    } as Object as CanvasRenderingContext2D;
    HTMLCanvasElement.prototype.getContext = () => mockCtx;
    graph = new Graph('TestApp');
    graph.nodes.push(new Node('Relu', ['X'], ['Y'], {}, 'ReluNode'));
  });

  it('app select edge', () => {
    const app = new ModifierApp({ container, initialGraph: graph });
    app.editor.selectEdge('X');
    expect(app.propsPanel.container.innerHTML).toContain('X');
  });

  afterEach(() => {
    document.body.innerHTML = '';
  });

  it('initializes fully', () => {
    const app = new ModifierApp({ container, initialGraph: graph });
    expect(app.graph).toBe(graph);
    expect(app.renderer).toBeDefined();
    expect(app.propsPanel).toBeDefined();

    // search by name
    const inputs = container.querySelectorAll('input');
    const nameSearch =
      Array.from(inputs).find(
        (i) => i.placeholder && i.placeholder.includes('Find Node by Name'),
      ) || inputs[1]!;
    nameSearch.value = graph.nodes[0]!.id;
    nameSearch.dispatchEvent(new Event('change'));
    expect(app.editor.selectedNodeIds.has(graph.nodes[0]!.id)).toBe(true);

    // search by type
    const typeSearch =
      Array.from(inputs).find(
        (i) => i.placeholder && i.placeholder.includes('Find Node by Type'),
      ) || inputs[2]!;
    typeSearch.value = 'Relu';
    typeSearch.dispatchEvent(new Event('change'));
    expect(app.editor.selectedNodeIds.has(graph.nodes[0]!.id)).toBe(true);

    // select empty
    app.editor.clearSelection();
    expect(app.propsPanel.container.innerHTML).toContain('Graph Properties');

    // Keyboard delete
    app.editor.selectNode(graph.nodes[0]!.id);
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Delete' }));
    expect(app.graph.nodes.length).toBe(0);
  });
});

// Mock roundRect for jsdom
const canvasMock = document.createElement('canvas');
const ctxMock = canvasMock.getContext('2d');
if (ctxMock && !ctxMock.roundRect) {
  CanvasRenderingContext2D.prototype.roundRect = function (x, y, w, h, r) {
    this.rect(x, y, w, h);
  };
}
