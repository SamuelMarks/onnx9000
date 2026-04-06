// @vitest-environment jsdom
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { Graph, Node } from '@onnx9000/core';
import { DagreLayoutEngine } from '../src/render/layout.js';
import { GraphRenderer } from '../src/render/canvas.js';

describe('GraphRenderer', () => {
  let originalCreateElement = document.createElement;
  let canvas: HTMLCanvasElement;
  let renderer: GraphRenderer;

  beforeEach(() => {
    canvas = document.createElement('canvas');
    canvas.width = 800;
    canvas.height = 600;

    // Mock getContext for JSDOM
    if (!originalCreateElement.name || !originalCreateElement.name.includes('bound '))
      originalCreateElement = document.createElement.bind(document);
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
      roundRect: vi.fn(),
      drawImage: vi.fn(),
    } as Object as CanvasRenderingContext2D;

    canvas.getContext = () => mockCtx as Object;
    document.createElement = (tag) => {
      const el = originalCreateElement(tag);
      if (tag === 'canvas') (el as Object).getContext = () => mockCtx;
      return el;
    };

    renderer = new GraphRenderer(canvas);
  });

  it('drawEdge hover highlighted', () => {
    const graph = new Graph('Test');
    const layout = {
      nodes: new Map(),
      edges: [{ sourceId: 'A', targetId: 'B', sourcePort: 'outA', targetPort: 'outA', path: [] }],
      bounds: { width: 10, height: 10 },
    };
    renderer.hoveredNodeId = 'A';
    expect(() => renderer.drawEdge(layout.edges[0], graph)).not.toThrow();
  });

  it('drawGroup edge cases', () => {
    const graph = new Graph('Test');
    const layout = new DagreLayoutEngine().compute(graph);
    expect(() => renderer.drawGroup(layout)).not.toThrow();

    graph.nodes.push(new Node('A', [], ['outA']));
    const layout2 = new DagreLayoutEngine().compute(graph);
    expect(() => renderer.drawGroup(layout2)).not.toThrow();
  });

  it('drawEdge shape', () => {
    const graph = new Graph('Test');
    graph.valueInfo.push({ name: 'outA', shape: [1], dtype: 'float32', id: '' });
    const layout = {
      nodes: new Map(),
      edges: [
        {
          sourceId: 'A',
          targetId: 'B',
          sourcePort: 'outA',
          targetPort: 'outA',
          path: [
            { x: 0, y: 0 },
            { x: 1, y: 1 },
          ],
        },
      ],
      bounds: { width: 10, height: 10 },
    };
    expect(() => renderer.drawEdge(layout.edges[0], graph)).not.toThrow();

    const layoutEmptyPath = {
      nodes: new Map(),
      edges: [{ sourceId: 'A', targetId: 'B', sourcePort: 'outA', targetPort: 'outA', path: [] }],
      bounds: { width: 10, height: 10 },
    };
    expect(() => renderer.drawEdge(layoutEmptyPath.edges[0], graph)).not.toThrow();
  });

  it('constructs and applies config', () => {
    expect(renderer).toBeDefined();
    expect(renderer.scale).toBe(1);
    expect(renderer.offsetX).toBe(0);
  });

  it('pans', () => {
    renderer.pan(10, 20);
    expect(renderer.offsetX).toBe(10);
    expect(renderer.offsetY).toBe(20);
  });

  it('zooms', () => {
    renderer.zoom(2, 400, 300);
    expect(renderer.scale).toBe(2);
    // Center was 400,300; it should adjust offset
    expect(renderer.offsetX).toBe(-400);
    expect(renderer.offsetY).toBe(-300);
    afterEach(() => {
      document.createElement = originalCreateElement;
    });
  });

  it('renders graph (mocked context)', () => {
    const graph = new Graph('Test');
    const nodeA = new Node('A', [], ['outA'], {}, 'NodeA');
    nodeA.opType = 'Constant';
    const nodeB = new Node('B', ['outA'], ['outB'], {}, 'NodeB');
    nodeB.opType = 'Add';
    graph.nodes.push(nodeA, nodeB);
    graph.outputs.push({ name: 'outB', shape: [], dtype: 'float32', id: '' });

    const layout = new DagreLayoutEngine().compute(graph);

    // We just verify it executes without errors on a JSDOM canvas
    expect(() => renderer.render(graph, layout)).not.toThrow();

    // Verify bounds extraction
    const bounds = renderer.getMinimapBounds(layout);
    expect(bounds.width).toBeGreaterThan(0);
  });

  it('fails if no context', () => {
    const badCanvas = document.createElement('canvas');
    badCanvas.getContext = () => null;
    expect(() => new GraphRenderer(badCanvas)).toThrow('Could not get 2D context');
  });

  it('draw node with selection', () => {
    const graph = new Graph('Test');
    const nodeA = new Node('A', [], ['outA'], {}, 'NodeA');
    graph.nodes.push(nodeA);
    const layout = new DagreLayoutEngine().compute(graph);
    renderer.selectedNodeIds.add(nodeA.id);
    renderer.hoveredNodeId = nodeA.id;

    expect(() => renderer.render(graph, layout)).not.toThrow();
  });
});
