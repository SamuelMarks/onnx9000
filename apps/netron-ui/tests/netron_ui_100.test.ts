import * as coreMod from '@onnx9000/core';
import { Attribute, Graph, Node, Onnx9000Error, Tensor, ValueInfo } from '@onnx9000/core';
import { describe, expect, it, vi } from 'vitest';
import { computeLayout } from '../src/layout/dag.js';
import { fetchAndParseModel } from '../src/parser/fetcher.js';
import { messageHandler } from '../src/parser/worker.js';
import { CanvasRenderer } from '../src/render/canvas.js';
import { SparsePrunerUI } from '../src/sparse_pruner.js';

describe('apps/netron-ui exhaustive 100% coverage', () => {
  it('covers parser/fetcher.ts and parser/worker.ts', async () => {
    const origFetch = globalThis.fetch;
    try {
      globalThis.fetch = vi
        .fn()
        .mockResolvedValueOnce({ ok: false, statusText: 'Not Found' })
        .mockResolvedValueOnce({
          ok: true,
          headers: { get: () => null },
          body: null,
        })
        .mockResolvedValueOnce({
          ok: true,
          headers: { get: () => null },
          body: {
            getReader: () => {
              let readCount = 0;
              return {
                read: async () => {
                  readCount++;
                  if (readCount === 1) return { done: false, value: undefined };
                  return { done: true, value: undefined };
                },
              };
            },
          },
        });

      await expect(
        fetchAndParseModel('https://github.com/org/repo/blob/main/m.onnx'),
      ).rejects.toThrow(Onnx9000Error);
      await expect(fetchAndParseModel('https://example.com/m.onnx')).rejects.toThrow(Onnx9000Error);
      await fetchAndParseModel('https://example.com/m.onnx');
    } finally {
      globalThis.fetch = origFetch;
    }

    const postMsg = vi.fn();
    await messageHandler(
      {
        data: {
          type: 'PARSE_BUFFER',
          get buffer() {
            throw 'non-error worker failure';
          },
        },
      } as unknown as MessageEvent,
      postMsg,
    );
    expect(postMsg).toHaveBeenCalledWith({
      type: 'PARSE_ERROR',
      error: 'non-error worker failure',
    });

    await messageHandler({ data: { type: 'UNKNOWN_TYPE' } } as MessageEvent, postMsg);

    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const parseSpy = vi.spyOn(coreMod, 'parseModelProto').mockResolvedValueOnce(new Graph('sab_g'));
    const sabBuf = new Uint8Array(new SharedArrayBuffer(8));
    await messageHandler(
      { data: { type: 'PARSE_BUFFER', buffer: sabBuf } } as MessageEvent,
      postMsg,
    );
    expect(logSpy).toHaveBeenCalledWith('Using SharedArrayBuffer for zero-copy parsing');
    parseSpy.mockRestore();
    logSpy.mockRestore();
  });

  it('covers layout/dag.ts TB and LR string constants, control Characters, and scopes', () => {
    const g = new Graph('dag_g');
    g.inputs.push(new ValueInfo('x_in', [1, 2], 'float32'));
    g.inputs.push(new ValueInfo('init_str_long', [1], 'string'));
    g.outputs.push(new ValueInfo('y_out', [1, 2], 'float32'));
    g.outputs.push(new ValueInfo('orphan_out', [1], 'float32'));
    g.initializers.push('init_str_long', 'init_str_short', 'init_f32');
    g.tensors.init_str_long = new Tensor(
      'init_str_long',
      [1],
      'string',
      true,
      false,
      new TextEncoder().encode('averylongstringconstantvalue\x00\x7f'),
    );
    g.tensors.init_str_short = new Tensor(
      'init_str_short',
      [1],
      'string',
      true,
      false,
      new TextEncoder().encode('short'),
    );
    g.tensors.init_f32 = new Tensor('init_f32', [2], 'float32', true, false, new Uint8Array(8));

    const nConstLong = new Node(
      'Constant',
      [],
      ['c_long'],
      {
        value_string: new Attribute('value_string', 'STRING', 'anotherverylongconstantstringvalue'),
      },
      'scopeA/subB/const_long',
    );
    nConstLong.id = 'n_const_long';
    const nConstShort = new Node(
      'Constant',
      ['', 'unproduced_tensor'],
      ['c_short'],
      { value_string: new Attribute('value_string', 'STRING', 'short_val') },
      'scopeA/subB/const_short',
    );
    nConstShort.id = 'n_const_short';
    const nAdd = new Node(
      'Add',
      ['x_in', 'init_f32', 'c_long', 'c_short'],
      ['y_out'],
      {},
      'scopeOnlyOne/add',
    );
    nAdd.id = 'n_add';

    g.nodes.push(nConstLong, nConstShort, nAdd);

    const tbLayout = computeLayout(g, 'TB');
    expect(tbLayout.groups?.length).toBeGreaterThan(0);

    const lrLayout = computeLayout(g, 'LR');
    expect(lrLayout.nodes.some((n) => n.stringValue?.endsWith('...'))).toBe(true);
  });

  it('covers render/canvas.ts touch, wheel, localStorage errors, regex, edge hover, and node variants', () => {
    const badCanvas = { getContext: () => null } as unknown as HTMLCanvasElement;
    expect(() => new CanvasRenderer(badCanvas)).toThrow(Onnx9000Error);

    if (typeof globalThis.Path2D === 'undefined') {
      (globalThis as unknown as { Path2D: unknown }).Path2D = class Path2D {
        moveTo() {}
        bezierCurveTo() {}
      };
    }

    const canvas = document.createElement('canvas');
    let pointInStroke = false;
    const mockCtx = {
      clearRect: vi.fn(),
      save: vi.fn(),
      restore: vi.fn(),
      translate: vi.fn(),
      scale: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      fill: vi.fn(),
      roundRect: vi.fn(),
      fillText: vi.fn(),
      fillRect: vi.fn(),
      measureText: () => ({ width: 50 }),
      setLineDash: vi.fn(),
      bezierCurveTo: vi.fn(),
      isPointInStroke: () => pointInStroke,
    };
    vi.spyOn(canvas, 'getContext').mockReturnValue(mockCtx as unknown as CanvasRenderingContext2D);

    const renderer = new CanvasRenderer(canvas);
    window.dispatchEvent(new Event('resize'));
    renderer.focusNode('missing');

    const layout = {
      width: 400,
      height: 400,
      groups: [
        { name: 'g_visible', depth: 1, x: 0, y: 0, width: 100, height: 100 },
        { name: 'g_offscreen', depth: 0, x: -99999, y: -99999, width: 10, height: 10 },
      ],
      nodes: [
        {
          id: 'n_in',
          opType: 'Input',
          name: 'in',
          type: 'input' as const,
          x: 0,
          y: 0,
          width: 100,
          height: 40,
        },
        {
          id: 'n_out',
          opType: 'Output',
          name: 'out',
          type: 'output' as const,
          x: 0,
          y: 50,
          width: 100,
          height: 40,
        },
        {
          id: 'n_const',
          opType: 'Initializer',
          name: 'const',
          type: 'constant' as const,
          stringValue: 'hi',
          x: 0,
          y: 100,
          width: 100,
          height: 40,
        },
        {
          id: 'n_cnode',
          opType: 'Constant',
          name: 'cnode',
          type: 'node' as const,
          stringValue: 'cval',
          x: 0,
          y: 150,
          width: 100,
          height: 40,
        },
        {
          id: 'n_q',
          opType: 'QuantizeLinear',
          name: '',
          type: 'node' as const,
          x: 0,
          y: 200,
          width: 100,
          height: 40,
        },
        {
          id: 'n_p',
          opType: 'Bitpack',
          name: 'bp',
          type: 'node' as const,
          x: 0,
          y: 250,
          width: 100,
          height: 40,
        },
        {
          id: 'n_add',
          opType: 'Add',
          name: 'add',
          type: 'node' as const,
          x: 0,
          y: 300,
          width: 100,
          height: 40,
        },
        {
          id: 'n_sub',
          opType: 'Sub',
          name: 'sub',
          type: 'node' as const,
          x: 0,
          y: 350,
          width: 100,
          height: 40,
        },
        {
          id: 'n_mul',
          opType: 'Mul',
          name: 'mul',
          type: 'node' as const,
          x: 0,
          y: 400,
          width: 100,
          height: 40,
        },
        {
          id: 'n_div',
          opType: 'Div',
          name: 'div',
          type: 'node' as const,
          x: 0,
          y: 450,
          width: 100,
          height: 40,
        },
        {
          id: 'n_off',
          opType: 'Relu',
          name: 'off',
          type: 'node' as const,
          x: -99999,
          y: -99999,
          width: 10,
          height: 10,
        },
      ],
      edges: [
        {
          from: 'n_in',
          to: 'n_out',
          tensorName: 't_f16',
          dtype: 'float16',
          shape: '[2, 4]',
          points: [
            { x: 0, y: 0 },
            { x: 0, y: 50 },
          ],
        },
        {
          from: 'n_in',
          to: 'n_out',
          tensorName: 't_i8',
          dtype: 'int8',
          shape: '[2, 4]',
          points: [
            { x: 0, y: 0 },
            { x: 100, y: 10 },
          ],
        },
        {
          from: 'n_in',
          to: 'n_out',
          tensorName: 't_i64',
          dtype: 'int64',
          shape: '[2, 4]',
          points: [
            { x: 0, y: 0 },
            { x: 100, y: 10 },
          ],
        },
        {
          from: 'n_in',
          to: 'n_out',
          tensorName: 't_bool',
          dtype: 'bool',
          shape: '[1]',
          points: [
            { x: 0, y: 0 },
            { x: 10, y: 100 },
          ],
        },
        {
          from: 'n_in',
          to: 'n_out',
          tensorName: 't_str',
          dtype: 'string',
          shape: '[N]',
          points: [
            { x: 0, y: 0 },
            { x: 10, y: 100 },
          ],
        },
        {
          from: 'n_in',
          to: 'n_out',
          tensorName: '',
          dtype: 'other',
          shape: '',
          points: [
            { x: 0, y: 0 },
            { x: 10, y: 100 },
          ],
        },
      ],
    };

    localStorage.setItem('onnxModifier_scale', '1');
    localStorage.setItem('onnxModifier_offsetX', '0');
    localStorage.setItem('onnxModifier_offsetY', '0');
    renderer.setLayout(layout);

    // Click when hoveredNode is null (deselects)
    (renderer as unknown as { hoveredNode: string | null }).hoveredNode = null;
    canvas.dispatchEvent(new MouseEvent('mousedown', { clientX: 500, clientY: 500 }));
    window.dispatchEvent(new MouseEvent('mouseup', { clientX: 500, clientY: 500 }));

    // Storage error branches via window.localStorage override
    const origStorageDesc = Object.getOwnPropertyDescriptor(window, 'localStorage');
    Object.defineProperty(window, 'localStorage', {
      configurable: true,
      value: {
        getItem: () => {
          throw new Error('storage blocked');
        },
        setItem: () => {
          throw new Error('storage blocked');
        },
      },
    });
    try {
      renderer.setLayout(layout);

      renderer.setSearchResults(['n_in']);
      renderer.setFilterControlEdges(true);
      renderer.setFilterControlEdges(false);
      renderer.setCustomColorRegex('');
      renderer.setCustomColorRegex('[invalid');
      renderer.setCustomColorRegex('^add$');
      renderer.focusNode('n_in');
      renderer.focusNode('non_existent');

      // Touch events (1 touch and 2 touches)
      const tStart1 = new Event('touchstart');
      Object.assign(tStart1, { touches: [{ clientX: 10, clientY: 20 }] });
      canvas.dispatchEvent(tStart1);
      const tMove1 = new Event('touchmove');
      Object.assign(tMove1, { touches: [{ clientX: 20, clientY: 30 }] });
      canvas.dispatchEvent(tMove1);
      canvas.dispatchEvent(new Event('touchend'));

      const tStartEmpty = new Event('touchstart');
      Object.assign(tStartEmpty, { touches: [] });
      canvas.dispatchEvent(tStartEmpty);

      // Mouse hover node, click, shift+click select & deselect, drag, and edge hover (for float16, int8, int64)
      (renderer as unknown as { offsetX: number; offsetY: number; scale: number }).offsetX = 0;
      (renderer as unknown as { offsetX: number; offsetY: number; scale: number }).offsetY = 0;
      (renderer as unknown as { offsetX: number; offsetY: number; scale: number }).scale = 1;

      window.dispatchEvent(new MouseEvent('mousemove', { clientX: 10, clientY: 10 }));
      canvas.dispatchEvent(new MouseEvent('mousedown', { clientX: 10, clientY: 10 }));
      window.dispatchEvent(new MouseEvent('mouseup', { clientX: 10, clientY: 10, shiftKey: true }));
      canvas.dispatchEvent(new MouseEvent('mousedown', { clientX: 10, clientY: 10 }));
      window.dispatchEvent(new MouseEvent('mouseup', { clientX: 10, clientY: 10, shiftKey: true }));

      // Edge hover when no node is hit
      pointInStroke = true;
      window.dispatchEvent(new MouseEvent('mousemove', { clientX: 500, clientY: 500 }));
      for (const edge of layout.edges) {
        (renderer as unknown as { hoveredEdge: unknown }).hoveredEdge = edge;
        renderer.render();
      }
      pointInStroke = false;

      // Drag with movement
      canvas.dispatchEvent(new MouseEvent('mousedown', { clientX: 10, clientY: 10 }));
      window.dispatchEvent(new MouseEvent('mousemove', { clientX: 30, clientY: 30 }));
      window.dispatchEvent(new MouseEvent('mouseup', { clientX: 30, clientY: 30 }));

      // Wheel zoom out to scale < 0.1
      canvas.dispatchEvent(new WheelEvent('wheel', { deltaY: 950 }));
    } finally {
      if (origStorageDesc) {
        Object.defineProperty(window, 'localStorage', origStorageDesc);
      }
    }
  });

  it('covers sparse_pruner.ts adjustSaliency, drag/drop yaml, runPruning without graph & with recipe, and getWeightDistribution', async () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    document.body.innerHTML = `
      <div id="log"></div>
      <input id="sparsity-slider" value="50" />
      <div id="sparsity-value"></div>
      <button id="run-btn"></button>
      <div id="drop-zone"></div>
      <div id="param-count"></div>
      <div id="current-sparsity"></div>
      <div id="est-speedup"></div>
      <div id="progress"></div>
      <div id="progress-fill"></div>
      <div id="progress-text"></div>
      <button id="download-btn"></button>
    `;
    const ui = new SparsePrunerUI();
    const slider = document.getElementById('sparsity-slider') as HTMLInputElement;
    slider.value = '75';
    slider.dispatchEvent(new Event('input'));
    expect(document.getElementById('sparsity-value')?.innerText).toBe('75%');

    const dz = document.getElementById('drop-zone') as HTMLElement;
    dz.dispatchEvent(new Event('dragover'));
    dz.dispatchEvent(new Event('dragleave'));
    dz.dispatchEvent(new Event('drop')); // no dataTransfer.files
    const yamlFile = {
      name: 'recipe.yaml',
      text: () => Promise.resolve('modifiers: []'),
    };
    const ymlFile = {
      name: 'recipe.yml',
      text: () => Promise.resolve('modifiers: []'),
    };
    const txtFile = {
      name: 'ignored.txt',
    };
    dz.dispatchEvent(
      Object.assign(new Event('drop'), { dataTransfer: { files: [yamlFile, ymlFile, txtFile] } }),
    );
    await new Promise((r) => setTimeout(r, 10));

    // runPruning when graph is null
    await ui.runPruning();

    ui.adjustSaliency('w', 0, 0.5);
    ui.updateStats();
    expect(ui.getWeightDistribution('w')).toEqual([]);

    const g = new Graph('sp_g');
    const tZero = new Tensor('w_zero', [2], 'float32', true, false, new Float32Array([0, 0]));
    const tNonZero = new Tensor('w_nz', [2], 'float32', true, false, new Float32Array([-1.0, 2.0]));
    const tNoData = new Tensor('w_nodata', [2], 'float32', true, false, null);
    g.tensors.w_zero = tZero;
    g.tensors.w_nz = tNonZero;
    g.tensors.w_nodata = tNoData;
    (ui as unknown as { graph: Graph }).graph = g;

    // runPruning with currentRecipe set using mocked setTimeout
    const origSetTimeout = globalThis.setTimeout;
    globalThis.setTimeout = ((cb: () => void) => {
      cb();
      return 0;
    }) as unknown as typeof setTimeout;
    try {
      await ui.runPruning();
    } finally {
      globalThis.setTimeout = origSetTimeout;
      delete (window as unknown as { currentRecipe?: string }).currentRecipe;
    }

    ui.adjustSaliency('missing', 0, 0.5);
    ui.adjustSaliency('w_nz', 0, 0.5);
    ui.adjustSaliency('w_nz', 5, 0.5);
    expect(ui.getWeightDistribution('missing')).toEqual([]);
    expect(ui.getWeightDistribution('w_nodata')).toEqual([]);
    expect(ui.getWeightDistribution('w_zero')).toHaveLength(20);
    expect(ui.getWeightDistribution('w_nz')).toHaveLength(20);
    logSpy.mockRestore();
  });
});
