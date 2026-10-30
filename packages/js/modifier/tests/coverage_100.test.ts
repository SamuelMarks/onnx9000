// @vitest-environment jsdom
import { Attribute, Graph, Node, Tensor, ValueInfo } from '@onnx9000/core';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
  __triggerCleanGraph,
  __triggerMakeDynamic,
  __triggerStripInitializers,
  ModifierApp,
} from '../src/app.js';
import { renderCustomEditor } from '../src/components/editors/custom_editors.js';
import { ModelExporter } from '../src/components/export/exporter.js';
import { InitializerInspector } from '../src/components/initializers/inspector.js';
import { PropertiesPanel } from '../src/components/properties.js';
import { Toolbar } from '../src/components/toolbar.js';
import { ModifierUtilities } from '../src/components/utilities.js';
import { GraphMutator } from '../src/GraphMutator.js';
import { GraphRenderer } from '../src/render/canvas.js';
import { DagreLayoutEngine } from '../src/render/layout.js';
import { parseRecipe } from '../src/sparse/modifier.js';
import { GraphEditor } from '../src/ui/editor.js';

describe('100% Coverage for @onnx9000/modifier', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.alert = vi.fn();
    window.confirm = vi.fn(() => true);
    window.prompt = vi.fn(() => 'my_annotation');
    URL.createObjectURL = vi.fn(() => 'blob:mock');
    URL.revokeObjectURL = vi.fn();
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined);
  });

  it('covers sparse/modifier.ts edge cases', () => {
    const mods = parseRecipe(`- !MagnitudePruningModifier
  : empty_key
  init_sparsity: 0.1`);
    expect(mods.length).toBe(1);
  });

  it('covers toolbar.ts strict checkbox', () => {
    const container = document.createElement('div');
    let strictVal = false;
    const noop = () => undefined;
    new Toolbar(container, {
      onCleanGraph: noop,
      onMakeDynamic: noop,
      onStripInitializers: noop,
      onFixMixedPrecision: noop,
      onRemoveTrainingNodes: noop,
      onFoldConstants: noop,
      onExtractWeights: noop,
      onSanitizeNames: noop,
      onValidateOpset: noop,
      onValidateGraph: noop,
      onAutoFix: noop,
      onDeduplicateConstants: noop,
      onExportStats: noop,
      onExportGraphJSON: noop,
      onFeedback: noop,
      onToggleStrict: (v) => {
        strictVal = v;
      },
      onSaveSession: noop,
      onExportModel: noop,
    });
    const cb = container.querySelector('input[type="checkbox"]') as HTMLInputElement;
    cb.checked = true;
    cb.onchange?.(new Event('change'));
    expect(strictVal).toBe(true);
  });

  it('covers render/layout.ts LR direction, pinned nodes, and timeout fallback', () => {
    const g = new Graph('layout_test');
    const n1 = new Node(
      'Relu',
      ['x'],
      ['y'],
      {
        pinned: new Attribute('pinned', 'INT', 1),
        pinned_x: new Attribute('pinned_x', 'INT', 100),
        pinned_y: new Attribute('pinned_y', 'INT', 200),
      },
      'n1',
    );
    const n2 = new Node(
      'Relu',
      ['y'],
      ['z'],
      {
        pinned: new Attribute('pinned', 'INT', 1),
      },
      'n2',
    );
    g.addNode(n1);
    g.addNode(n2);

    const lrEngine = new DagreLayoutEngine('LR');
    const layout = lrEngine.compute(g);
    expect(layout.edges.length).toBe(1);
    expect(layout.nodes.get(n1.id)?.position.x).toBe(120);

    // Timeout fallback (> 2000ms)
    vi.spyOn(performance, 'now').mockReturnValueOnce(0).mockReturnValueOnce(3000);
    const fallbackLayout = lrEngine.compute(g);
    expect(fallbackLayout.edges.length).toBe(0);
    vi.mocked(performance.now).mockRestore();
  });

  it('covers render/canvas.ts methods and branches', () => {
    const badCanvas = document.createElement('canvas');
    vi.spyOn(badCanvas, 'getContext').mockReturnValueOnce(null);
    expect(() => new GraphRenderer(badCanvas)).toThrow('Could not get 2D context');

    let useRoundRect = true;
    const makeFakeCtx = () =>
      ({
        save: vi.fn(),
        restore: vi.fn(),
        setTransform: vi.fn(),
        fillRect: vi.fn(),
        strokeRect: vi.fn(),
        beginPath: vi.fn(),
        moveTo: vi.fn(),
        lineTo: vi.fn(),
        quadraticCurveTo: vi.fn(),
        closePath: vi.fn(),
        fill: vi.fn(),
        stroke: vi.fn(),
        fillText: vi.fn(),
        setLineDash: vi.fn(),
        drawImage: vi.fn(),
        roundRect: useRoundRect ? vi.fn() : undefined,
        rect: vi.fn(),
      }) as unknown as CanvasRenderingContext2D;

    const origCreateElement = document.createElement.bind(document);
    vi.spyOn(document, 'createElement').mockImplementation((tag: string) => {
      const el = origCreateElement(tag);
      if (tag === 'canvas') {
        vi.spyOn(el as HTMLCanvasElement, 'getContext').mockImplementation(
          () => makeFakeCtx() as never,
        );
      }
      return el;
    });

    const canvas = document.createElement('canvas');
    canvas.width = 400;
    canvas.height = 300;
    const renderer = new GraphRenderer(canvas);

    const g = new Graph('canvas_g');
    const n1 = new Node(
      'NonMaxSuppression',
      ['x'],
      ['y'],
      { _inference_path_active: new Attribute('_inference_path_active', 'INT', 1) },
      'n1',
    );
    const n2 = new Node('Constant', [], ['c'], {}, 'n2');
    g.addNode(n1);
    g.addNode(n2);
    g.outputs.push(new ValueInfo('y', [1, 4], 'float32'));
    g.valueInfo.push(new ValueInfo('y', [1, 4], 'float32'));

    const layout = new DagreLayoutEngine('TB').compute(g);
    layout.edges.push({
      sourceId: n1.id,
      targetId: n2.id,
      sourcePort: 'y',
      targetPort: 'y',
      path: [
        { x: 0, y: 0 },
        { x: 10, y: 10 },
      ],
    });
    layout.edges.push({
      sourceId: 'other',
      targetId: 'other2',
      sourcePort: 'missing',
      targetPort: 'missing',
      path: [],
    });

    renderer.hoveredNodeId = n1.id;
    renderer.selectedNodeIds.add(n1.id);
    renderer.render(g, layout);
    // Render again to hit nodeCache hit branch
    renderer.render(g, layout);

    // Also test without roundRect on offscreen context
    useRoundRect = false;
    const n3 = new Node('TopK', ['x'], ['k'], {}, 'n3');
    g.addNode(n3);
    const layout2 = new DagreLayoutEngine('TB').compute(g);
    renderer.hoveredNodeId = null;
    renderer.selectedNodeIds.clear();
    renderer.render(g, layout2);

    renderer.pan(10, 20);
    renderer.zoom(2, 100, 100);
    expect(renderer.getMinimapBounds(layout2).width).toBeGreaterThan(0);
    renderer.destroy();
    expect(canvas.width).toBe(0);
  });

  it('covers components/utilities.ts all methods', () => {
    const g = new Graph('util_g');
    g.inputs.push(new ValueInfo('x', [1, 4], 'float32'));
    g.outputs.push(new ValueInfo('z', [1, 4], 'float32'));
    g.valueInfo.push(new ValueInfo('y', [1, 4], 'float32'));
    g.valueInfo.push(new ValueInfo('scalar', [], 'float32'));
    const t = new Tensor('w', [1], 'float32', true, false, new Float32Array([1]));
    g.tensors.w = t;
    g.initializers.push('w');

    const n1 = new Node('Relu', ['x', ''], ['y', ''], {}, 'old_relu');
    const n2 = new Node('Relu', ['y', 'missing_init'], ['z'], {}, '');
    g.addNode(n1);
    g.addNode(n2);

    const mutator = new GraphMutator(g);
    const utils = new ModifierUtilities(mutator);

    utils.makeDynamic();
    expect(g.inputs[0]?.shape[0]).toBe('batch_size');

    utils.insertIdentity('non_existent_edge');
    utils.insertIdentity('y');
    mutator.undo();
    mutator.redo();

    utils.regexRenameNodes('^old_', 'new_');
    utils.regexRenameNodes('nomatch', 'x');
    utils.autoFormatNodeNames();

    const sub = utils.extractSubgraph([n1.id, n2.id]);
    expect(sub.nodes.length).toBe(2);

    // autoFixMissingInitializers when missing exists, then when none missing
    utils.autoFixMissingInitializers();
    utils.autoFixMissingInitializers();

    // validateOpset: unset, out of range, in range
    expect(utils.validateOpset()).toBe(true);
    utils.changeOpsetVersion('', 10);
    expect(utils.validateOpset()).toBe(false);
    utils.changeOpsetVersion('', 17);
    expect(utils.validateOpset()).toBe(true);

    expect(utils.injectCastNode('x', 'FLOAT')).toBe('x_casted');
    expect(utils.injectCastNode('x', 'INT64')).toBe('x_casted');

    utils.stripInitializers();
    expect(g.initializers.length).toBe(0);
  });

  it('covers components/export/exporter.ts all methods', async () => {
    const g = new Graph('exp_g');
    g.inputs.push(new ValueInfo('x', [1, 'dyn'], 'float32'));
    g.outputs.push(new ValueInfo('y', [1, 'dyn'], 'float32'));
    const n1 = new Node('Relu', ['x'], ['y'], {}, '');
    g.addNode(n1);
    const t = new Tensor('w', [2, 'dyn'], 'float32', true, false, new Float32Array([1, 2]));
    g.tensors.w = t;
    g.initializers.push('w');
    g.initializers.push('missing_t');

    const mutator = new GraphMutator(g);
    const exporter = new ModelExporter(mutator);

    exporter.downloadBlob('test.onnx', new Uint8Array([1, 2]));
    mutator.addNode('Identity', ['y'], ['y2'], {}, 'id1');
    expect(exporter.generateEditLog()).toContain('mutation_applied');
    expect(exporter.generatePythonHelperScript()).toContain('helper.make_node');
    expect(exporter.generateSummary()).toContain('Parameters: 2');
    expect(exporter.generateGraphvizDot()).toContain('digraph G');
    exporter.exportStatsCSV();
    expect(exporter.promptChangesBeforeExport()).toBe(true);

    exporter.saveSessionToLocalStorage();
    const origStorage = window.localStorage;
    Object.defineProperty(window, 'localStorage', { value: null, configurable: true });
    exporter.saveSessionToLocalStorage();
    Object.defineProperty(window, 'localStorage', { value: origStorage, configurable: true });

    // Memory threshold > 1000MB
    const bigTensor = {
      name: 'big',
      shape: [1],
      dtype: 'float32',
      data: { byteLength: 1001 * 1024 * 1024 },
    } as unknown as Tensor;
    g.tensors.big = bigTensor;
    g.initializers.push('big');
    vi.spyOn(window, 'confirm').mockReturnValueOnce(false);
    await expect(exporter.exportModel()).rejects.toThrow('Export aborted');
  });

  it('covers components/initializers/inspector.ts all methods', async () => {
    const fakeCtx = {
      fillRect: vi.fn(),
      strokeRect: vi.fn(),
    } as unknown as CanvasRenderingContext2D;
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(fakeCtx as never);

    const g = new Graph('insp_g');
    const mutator = new GraphMutator(g);
    const container = document.createElement('div');
    const inspector = new InitializerInspector(container, mutator);

    // 1. No data
    inspector.render(new Tensor('empty', [1], 'float32', true, false, null));
    expect(container.textContent).toContain('No internal data');

    // 2. Unsupported dtype
    inspector.render(new Tensor('fp16', [1], 'float16', true, false, new Uint16Array([1])));
    expect(container.textContent).toContain('Unsupported view');

    // 3. Supported dtypes: float64, int32, uint8
    const t64 = new Tensor('f64', [1], 'float64', true, false, new Float64Array([1.5]));
    g.tensors.f64 = t64;
    inspector.render(t64);
    // Scalar editor input change
    const scalarInput = container.querySelector('input[type="number"]') as HTMLInputElement;
    scalarInput.value = '2.5';
    scalarInput.onchange?.({ target: scalarInput } as unknown as Event);
    scalarInput.value = 'not_a_num';
    scalarInput.onchange?.({ target: scalarInput } as unknown as Event);

    // Click all action buttons on t64 (non-float32, so Cast to FP32 works!)
    const buttons = Array.from(container.querySelectorAll('button'));
    for (const btn of buttons) {
      btn.click();
    }
    mutator.undo();
    mutator.redo();
    mutator.undo(); // revert back to float64 so Upload .bin with 8-byte buffer matches float64

    // Test Upload .bin onchange
    const origCreate = document.createElement.bind(document);
    let createdFileInput: HTMLInputElement | null = null;
    vi.spyOn(document, 'createElement').mockImplementation((tag: string) => {
      const el = origCreate(tag);
      if (tag === 'input') {
        createdFileInput = el as HTMLInputElement;
        vi.spyOn(el, 'click').mockImplementation(() => undefined);
      }
      return el;
    });
    const uploadBtn = buttons.find((b) => b.textContent === 'Upload .bin') as HTMLButtonElement;
    uploadBtn.click();
    if (createdFileInput) {
      const fakeFile = {
        arrayBuffer: async () => new Float64Array([9.0]).buffer,
      };
      await (createdFileInput as HTMLInputElement).onchange?.({
        target: { files: [fakeFile] },
      } as unknown as Event);
      await (createdFileInput as HTMLInputElement).onchange?.({
        target: { files: [] },
      } as unknown as Event);
    }
    vi.mocked(document.createElement).mockRestore();

    // 4. int32 & uint8 & 2D heatmap
    const tInt32 = new Tensor('i32', [2, 2], 'int32', true, false, new Int32Array([0, 1, 2, 3]));
    g.tensors.i32 = tInt32;
    inspector.render(tInt32);
    // Also test noise and prune on 0/small values
    const i32Btns = Array.from(container.querySelectorAll('button'));
    for (const b of i32Btns) b.click();

    const tU8 = new Tensor('u8', [20, 20], 'uint8', true, false, new Uint8Array(400));
    inspector.render(tU8);

    const tShort = new Tensor('short', [2, 2], 'float32', true, false, new Float32Array([1]));
    inspector.render(tShort);

    // Heatmap when getContext('2d') returns null
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValueOnce(null);
    inspector.render(tInt32);
  });

  it('covers ui/editor.ts all methods', () => {
    const g = new Graph('ed_g');
    const n1 = new Node('Relu', ['x'], ['y'], {}, 'n1');
    const n2 = new Node('Relu', ['y'], ['z'], {}, '');
    g.addNode(n1);
    g.addNode(n2);

    const mutator = new GraphMutator(g);
    const editor = new GraphEditor(g, mutator);

    editor.selectNode(n1.id, false);
    editor.selectNode(n2.id, true);
    editor.selectEdge('y', false);
    editor.selectEdge('z', true);

    editor.alignNodes('Left');
    editor.selectNode(n1.id, false);
    editor.alignNodes('Center');
    editor.snapToGrid(20);
    editor.snapToGrid();
    editor.pinNodes();
    editor.unpinNodes();

    editor.disconnectNode('non_existent');
    editor.disconnectNode(n1.id);
    mutator.undo();
    mutator.redo();

    editor.duplicateNode('non_existent');
    editor.duplicateNode(n2.id);

    editor.duplicateSubgraph([]);
    editor.duplicateSubgraph([n1.id, n2.id]);
    mutator.undo();
    mutator.redo();

    editor.connectPorts('x', 'non_existent', 0);
    editor.connectPorts('x', n2.id, 0);
    mutator.undo();
    mutator.redo();
    editor.connectPorts('x', n2.id, 5);
    mutator.undo();

    // deleteSelection with edge and node
    editor.selectNode(n1.id, false);
    editor.selectEdge('y', true);
    editor.deleteSelection();
    mutator.undo();
    mutator.redo();
  });

  it('covers components/editors/custom_editors.ts all editors', () => {
    const g = new Graph('ce_g');
    const mutator = new GraphMutator(g);
    const rootContainer = document.createElement('div');

    const ops = [
      new Node(
        'Conv',
        ['x', 'w'],
        ['y'],
        {
          strides: new Attribute('strides', 'INTS', [1, 1]),
          group: new Attribute('group', 'INT', 1),
        },
        '',
      ),
      new Node(
        'Gemm',
        ['a', 'b'],
        ['c'],
        {
          transA: new Attribute('transA', 'INT', 1),
          alpha: new Attribute('alpha', 'FLOAT', 1.0),
        },
        '',
      ),
      new Node('Split', ['x'], ['y'], {}, 'split'),
      new Node('Resize', ['x'], ['y'], { mode: new Attribute('mode', 'STRING', 'linear') }, ''),
      new Node('Squeeze', ['x'], ['y'], { axes: new Attribute('axes', 'INTS', [0]) }, 'sq1'),
      new Node('Unsqueeze', ['x'], ['y'], {}, 'unsq1'),
      new Node('Cast', ['x'], ['y'], { to: new Attribute('to', 'INT', 1) }, ''),
      new Node('Cast', ['x'], ['y'], {}, 'cast2'),
      new Node('Constant', [], ['y'], {}, 'const_empty'),
      new Node('Constant', [], ['y'], { value: new Attribute('value', 'INT', 42) }, 'const_scalar'),
      new Node(
        'Constant',
        [],
        ['y'],
        { value: new Attribute('value', 'TENSOR', { data: {} } as unknown as Tensor) },
        'const_bad_tensor',
      ),
      new Node(
        'Constant',
        [],
        ['y'],
        {
          value: new Attribute(
            'value',
            'TENSOR',
            new Tensor('t', [150], 'float32', true, false, new Float32Array(150)),
          ),
        },
        'const_paged',
      ),
      new Node(
        'If',
        ['cond'],
        ['y'],
        {
          then_branch: new Attribute('then_branch', 'GRAPH', new Graph('then')),
          else_branch: new Attribute('else_branch', 'INT', 0),
        },
        '',
      ),
      new Node(
        'Loop',
        ['m', 'c'],
        ['y'],
        { body: new Attribute('body', 'GRAPH', new Graph('body')) },
        'loop_node',
      ),
      new Node('SequenceConstruct', ['x'], ['y'], {}, 'seq_node'),
      new Node(
        'ScatterND',
        ['d', 'i', 'u'],
        ['y'],
        { reduction: new Attribute('reduction', 'STRING', 'add') },
        'scatter1',
      ),
      new Node('ScatterND', ['d', 'i', 'u'], ['y'], {}, 'scatter2'),
      new Node('CastMap', ['x'], ['y'], {}, 'map_node'),
    ];

    for (const node of ops) {
      g.addNode(node);
      const subContainer = document.createElement('div');
      rootContainer.appendChild(subContainer);
      expect(renderCustomEditor({ container: subContainer, node, mutator })).toBe(true);
    }

    // Trigger all inputs, selects, checkboxes, and buttons inside rootContainer
    const inputs = Array.from(rootContainer.querySelectorAll('input'));
    for (const inp of inputs) {
      if (inp.type === 'checkbox') {
        inp.checked = !inp.checked;
        inp.dispatchEvent(new Event('change'));
        inp.checked = !inp.checked;
        inp.dispatchEvent(new Event('change'));
      } else if (inp.type === 'number') {
        inp.value = '2.5';
        inp.dispatchEvent(new Event('change'));
        inp.value = 'invalid';
        inp.dispatchEvent(new Event('change'));
      } else {
        inp.value = '[1, 2]';
        inp.dispatchEvent(new Event('change'));
        inp.value = '"not_array"';
        inp.dispatchEvent(new Event('change'));
        inp.value = '{invalid_json';
        inp.dispatchEvent(new Event('change'));
      }
    }

    const selects = Array.from(rootContainer.querySelectorAll('select'));
    for (const sel of selects) {
      sel.value = '2'; // triggers low-precision integer warning in Cast editor
      sel.dispatchEvent(new Event('change'));
      sel.value = '1';
      sel.dispatchEvent(new Event('change'));
    }

    const buttons = Array.from(rootContainer.querySelectorAll('button'));
    for (const btn of buttons) {
      btn.click();
    }
    for (const btn of buttons.reverse()) {
      btn.click();
    }
  });

  it('covers components/properties.ts all branches', () => {
    const g = new Graph('prop_g');
    g.docString = undefined as unknown as string;
    const n1 = new Node(
      'Relu',
      ['', 'x'],
      ['', 'y'],
      {
        ints_attr: new Attribute('ints_attr', 'INTS', [1, 2]),
        int_attr: new Attribute('int_attr', 'INT', 1),
        float_attr: new Attribute('float_attr', 'FLOAT', 1.5),
      },
      '',
    );
    (n1 as unknown as Record<string, number>)._profiling_time_ms = 1.23;
    const nConv = new Node('Conv', ['x', 'w'], ['y'], {}, 'conv_custom');
    g.addNode(n1);
    g.addNode(nConv);
    g.inputs.push(new ValueInfo('x', [1, 4], 'seq_float32'));
    const t = new Tensor('w', [1], 'float32', true, false, new Float32Array([1]));
    g.tensors.w = t;

    const mutator = new GraphMutator(g);
    const container = document.createElement('div');
    const panel = new PropertiesPanel(container, mutator);

    // Custom editor early return
    panel.renderNode(nConv, g);

    panel.renderNode(n1, g);
    const inputs = Array.from(container.querySelectorAll('input'));
    for (const inp of inputs) {
      inp.value = '[3, 4]';
      inp.onchange?.({ target: inp } as unknown as Event);
      inp.value = 'not_json';
      inp.onchange?.({ target: inp } as unknown as Event);
      inp.value = '10';
      inp.onchange?.({ target: inp } as unknown as Event);
      inp.value = '';
      inp.onchange?.({ target: inp } as unknown as Event);
    }

    // Click Export Node & Copy Attributes
    const btns = Array.from(container.querySelectorAll('button'));
    for (const b of btns) b.click();

    // Re-render node to show & click Paste Attributes
    panel.renderNode(n1, g);
    const pasteBtn = Array.from(container.querySelectorAll('button')).find(
      (b) => b.textContent === 'Paste Attributes',
    );
    pasteBtn?.click();

    // Render edge: tensor, sequence valueInfo, unknown edge
    panel.renderEdge('w', g);
    panel.renderEdge('x', g);
    const edgeInputs = Array.from(container.querySelectorAll('input'));
    if (edgeInputs[0]) {
      edgeInputs[0].value = 'batch';
      edgeInputs[0].dispatchEvent(new Event('change'));
      edgeInputs[0].value = '8';
      edgeInputs[0].dispatchEvent(new Event('change'));
    }
    panel.renderEdge('y', g);

    // Render graph properties with undefined metadataProps
    (g as unknown as Record<string, unknown>).metadataProps = undefined;
    panel.renderGraphProperties(g);
    const graphInputs = Array.from(container.querySelectorAll('input'));
    for (const inp of graphInputs) {
      inp.value = '123';
      inp.dispatchEvent(new Event('change'));
      inp.value = 'abc';
      inp.dispatchEvent(new Event('change'));
    }
    const addMetaBtn = container.querySelector('button') as HTMLButtonElement;
    addMetaBtn.click();
    const addMetaBtn2 = container.querySelector('button') as HTMLButtonElement;
    addMetaBtn2.click();
    window.prompt = vi.fn(() => null);
    const addMetaBtn3 = container.querySelector('button') as HTMLButtonElement;
    addMetaBtn3.click();
  });

  it('covers GraphMutator.ts remaining branches', () => {
    const g = new Graph('mut_g');
    g.inputs.push(new ValueInfo('x', [1, 2], 'float32'));
    g.inputs.push(new ValueInfo('w', [2], 'float32'));
    g.outputs.push(new ValueInfo('z', [1, 2], 'float32'));
    g.valueInfo.push(new ValueInfo('y', [1, 2], 'float32'));
    const tW = new Tensor('w', [2], 'float32', true, false, new Float32Array([1, 2]));
    g.tensors.w = tW;
    g.initializers.push('w');

    const n1 = new Node('Relu', ['x'], ['y'], {}, 'n1');
    const n2 = new Node('Dropout', ['y'], ['z', 'mask'], {}, 'n2');
    const n3 = new Node('Add', ['z', 'other_in'], ['final_out'], {}, '');
    g.outputs.push(new ValueInfo('mask', [1, 2], 'bool'));
    g.addNode(n1);
    g.addNode(n2);
    g.addNode(n3);

    const mutator = new GraphMutator(g);

    // removeNode: non-existent, unnamed node, and numeric index with healEdges = true, then undo
    mutator.removeNode('non_existent');
    mutator.removeNode(n3.id);
    mutator.undo();
    mutator.removeNode(0, true);
    mutator.undo();

    // renameNode then undo
    mutator.renameNode('n1', 'n1_new');
    mutator.undo();

    // replaceNode then undo & redo & undo, and non-existent
    mutator.replaceNode('non_existent', new Node('Relu', ['x'], ['y'], {}, 'n1'));
    mutator.replaceNode('n1', new Node('Sigmoid', ['x'], ['y'], {}, 'n1'));
    mutator.undo();
    mutator.redo();
    mutator.undo();

    // changeNodeOpType then undo
    mutator.changeNodeOpType('n1', 'Tanh');
    mutator.undo();

    // setNodeAttribute & removeNodeAttribute then undo
    mutator.setNodeAttribute('n1', 'alpha', 1.0, 'FLOAT');
    mutator.removeNodeAttribute('n1', 'alpha');
    mutator.undo();

    // updateMetadata with defined and undefined args then undo
    mutator.updateMetadata('prod', '2.0', 'doc');
    mutator.undo();
    mutator.updateMetadata(undefined, undefined, undefined);
    mutator.undo();

    // overrideShape on input, output, valueInfo, and brand new tensorName then undo
    mutator.overrideShape('x', [2, 2], 'float32');
    mutator.undo();
    mutator.overrideShape('z', [2, 2], 'float32');
    mutator.undo();
    mutator.overrideShape('y', [2, 2], 'float32');
    mutator.undo();
    mutator.overrideShape('brand_new_vi', [2, 2], 'float32');
    mutator.undo();

    // extractSubgraph with initializer input ('w', 'missing_init_in_inputs') and empty input/output strings
    g.initializers.push('init_no_vi');
    n1.inputs.push('w', 'init_no_vi', '');
    n1.outputs.push('');
    const sub = mutator.extractSubgraph([n1.id]);
    expect(sub.nodes.length).toBe(1);
    n1.inputs.splice(1);
    n1.outputs.pop();
    g.initializers.pop();
    const sub2 = mutator.extractSubgraph([n2.id]);
    expect(sub2.outputs.length).toBeGreaterThan(0);

    // removeInitializer then undo
    mutator.removeInitializer('w');
    mutator.undo();

    // removeInput when used vs unused
    mutator.removeInput('x');
    mutator.addInput('unused_in', 'float32', [1]);
    mutator.removeInput('unused_in');
    mutator.undo();
    mutator.redo();
    mutator.removeInput('non_existent');

    // addOutput inferring from valueInfo, initializer, or fallback
    mutator.addOutput('y');
    mutator.addOutput('w');
    mutator.addOutput('unknown_edge');
    mutator.undo();
    mutator.redo();

    // removeOutput
    mutator.removeOutput('non_existent');
    mutator.removeOutput('unknown_edge');
    mutator.undo();
    mutator.redo();

    // updateInitializer dtype branches: float16, int8, int32, int64, other
    for (const [dtype, bytes] of [
      ['float16', 2],
      ['int8', 1],
      ['int32', 4],
      ['int64', 8],
      ['uint8', 1],
    ] as const) {
      g.tensors[`t_${dtype}`] = new Tensor(
        `t_${dtype}`,
        [1, 'dyn'],
        dtype,
        true,
        false,
        new Uint8Array(bytes),
      );
      mutator.updateInitializer(`t_${dtype}`, new Uint8Array(bytes));
      mutator.undo();
    }
    expect(() => mutator.updateInitializer('w', new Uint8Array(3))).toThrow();
    mutator.updateInitializer('non_existent', new Uint8Array(1));

    // convertInputToInitializer & convertInitializerToInput
    mutator.convertInputToInitializer('non_existent', new Float32Array([1]));
    mutator.convertInputToInitializer('x', new Float32Array([1, 2]));
    mutator.undo();
    mutator.redo();

    mutator.convertInitializerToInput('non_existent');
    mutator.convertInitializerToInput('w');
    mutator.undo();
    mutator.redo();

    // topologicalSort with cycle
    const c1 = new Node('Identity', ['cy2'], ['cy1'], {}, 'c1');
    const c2 = new Node('Identity', ['cy1'], ['cy2'], {}, 'c2');
    g.addNode(c1);
    g.addNode(c2);
    mutator.topologicalSort();
    mutator.undo();

    // inferShapesGlobally
    mutator.inferShapesGlobally();

    // cleanGraph, fixMixedPrecision, removeTrainingNodes, foldConstants, extractWeights, sanitizeNames, deduplicateConstants
    const castNode = new Node('Cast', ['x'], ['cx'], { to: new Attribute('to', 'INT', 1) }, 'cast');
    g.addNode(castNode);
    mutator.fixMixedPrecision('FLOAT16');
    mutator.fixMixedPrecision('FLOAT');
    mutator.undo();

    mutator.removeTrainingNodes();
    mutator.undo();
    mutator.redo();
    mutator.removeTrainingNodes(); // second call hits nodesToRemove.length === 0

    const bigConstTensor = new Tensor(
      'big_c',
      [500],
      'float32',
      true,
      false,
      new Float32Array(500),
    );
    const nullDataTensor = new Tensor('null_c', [1], 'float32', true, false, null);
    const smallConstTensor = new Tensor(
      'small_c',
      [2],
      'float32',
      true,
      false,
      new Float32Array([1, 2]),
    );
    g.addNode(
      new Node(
        'Constant',
        [],
        ['c_out1'],
        { value: new Attribute('value', 'TENSOR', smallConstTensor) },
        'c_small1',
      ),
    );
    g.addNode(
      new Node(
        'Constant',
        [],
        ['c_out2'],
        { value: new Attribute('value', 'TENSOR', smallConstTensor) },
        'c_small2',
      ),
    );
    g.addNode(
      new Node(
        'Constant',
        [],
        ['c_out_big'],
        { value: new Attribute('value', 'TENSOR', bigConstTensor) },
        'c_big',
      ),
    );
    g.addNode(
      new Node(
        'Constant',
        [],
        ['c_out_null'],
        { value: new Attribute('value', 'TENSOR', nullDataTensor) },
        'c_null',
      ),
    );
    g.addNode(
      new Node('Constant', [], ['c_scalar1'], { value: new Attribute('value', 'INT', 7) }, 'cs1'),
    );
    g.addNode(
      new Node('Constant', [], ['c_scalar2'], { value: new Attribute('value', 'INT', 7) }, 'cs2'),
    );
    g.addNode(new Node('Constant', [], ['c_no_attr'], {}, 'c_no_attr'));
    g.addNode(new Node('Add', ['c_out1', 'c_out2'], ['c_sum'], {}, 'add_c'));

    // Call deduplicateConstants FIRST while Tensor instances are still real Tensor instances!
    mutator.deduplicateConstants();
    mutator.undo();

    // Re-set real Tensor instances on c_big and c_null before extractWeights
    const cBigNode = g.nodes.find((n) => n.name === 'c_big') as Node;
    cBigNode.attributes.value = new Attribute('value', 'TENSOR', bigConstTensor);
    const cNullNode = g.nodes.find((n) => n.name === 'c_null') as Node;
    cNullNode.attributes.value = new Attribute('value', 'TENSOR', nullDataTensor);

    mutator.extractWeights(100);
    mutator.undo();
    mutator.redo();

    mutator.foldConstants();
    mutator.undo();
    mutator.redo();

    g.tensors.unmapped_tensor = smallConstTensor;
    mutator.sanitizeNames();
    mutator.undo();
    mutator.redo();

    mutator.cleanGraph();
    mutator.undo();
    mutator.redo();
    mutator.cleanGraph(); // second call hits toRemove.length === 0
  });

  it('covers app.ts toolbar callbacks, search, and helper triggers', async () => {
    const fakeCtx = {
      save: vi.fn(),
      restore: vi.fn(),
      setTransform: vi.fn(),
      fillRect: vi.fn(),
      strokeRect: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      quadraticCurveTo: vi.fn(),
      closePath: vi.fn(),
      fill: vi.fn(),
      stroke: vi.fn(),
      fillText: vi.fn(),
      setLineDash: vi.fn(),
      drawImage: vi.fn(),
      roundRect: vi.fn(),
      rect: vi.fn(),
    } as unknown as CanvasRenderingContext2D;
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(fakeCtx as never);

    const openSpy = vi.spyOn(window, 'open').mockReturnValue({
      document: { write: vi.fn() },
    } as unknown as Window);

    // Test localStorage throwing in constructor
    const origDesc = Object.getOwnPropertyDescriptor(window, 'localStorage');
    Object.defineProperty(window, 'localStorage', {
      get() {
        throw new Error('storage blocked');
      },
      configurable: true,
    });

    const g = new Graph('app_g');
    g.opsetImports[''] = 11; // triggers deprecated opset banner
    g.inputs.push(new ValueInfo('x', [1, 4], 'float32'));
    g.outputs.push(new ValueInfo('y', [1, 4], 'float32'));
    const n1 = new Node('Relu', ['x'], ['y'], {}, 'relu_node');
    g.addNode(n1);

    const container = document.createElement('div');
    document.body.appendChild(container);
    const app = new ModifierApp({ container, initialGraph: g });
    if (origDesc) {
      Object.defineProperty(window, 'localStorage', origDesc);
    }

    __triggerCleanGraph(app);
    __triggerMakeDynamic(app);
    __triggerStripInitializers(app);

    // Click all toolbar buttons when graph is valid
    const buttons = Array.from(container.querySelectorAll('button'));
    for (const btn of buttons) {
      btn.click();
    }

    // Select non-existent node, edge, & clear selection
    app.editor.selectNode('non_existent_id');
    app.editor.selectEdge('x');
    app.editor.clearSelection();

    // Toggle strict mode checkbox
    const cb = container.querySelector('input[type="checkbox"]') as HTMLInputElement;
    cb.checked = true;
    cb.onchange?.(new Event('change'));
    cb.checked = false;
    cb.onchange?.(new Event('change'));

    // Search by Name (matching by id as well as name) and Search by Type
    const searchName = container.querySelector(
      'input[placeholder="Search by Name (74)"]',
    ) as HTMLInputElement;
    searchName.value = n1.id;
    searchName.onchange?.({ target: searchName } as unknown as Event);
    searchName.value = 'missing';
    searchName.onchange?.({ target: searchName } as unknown as Event);

    const searchType = container.querySelector(
      'input[placeholder="Search by Type (75)"]',
    ) as HTMLInputElement;
    searchType.value = 'Relu';
    searchType.onchange?.({ target: searchType } as unknown as Event);

    // Test keydown events: Backspace when INPUT focused vs not focused, and Ctrl+D / Meta+D when selection empty vs non-empty
    searchName.focus();
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Backspace' }));
    searchName.blur();
    app.editor.clearSelection();
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'd' }));
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'd', ctrlKey: true }));
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'd', metaKey: true }));
    app.editor.selectNode(n1.id);
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'd', ctrlKey: true }));
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Delete' }));

    // Make graph invalid and click Validate Graph & Export Model
    app.graph.addNode(new Node('Add', ['missing1', 'missing2'], ['bad_out'], {}, 'bad'));
    const validateBtn = buttons.find(
      (b) => b.textContent === 'Validate Graph',
    ) as HTMLButtonElement;
    validateBtn.click();
    const exportBtn = buttons.find((b) => b.textContent === 'Export Model') as HTMLButtonElement;
    exportBtn.click();

    // Trigger _updateViewInternal and test pb.parentNode being null vs present
    (app as unknown as { _updateViewInternal: () => void })._updateViewInternal();
    app.centerPanel.innerHTML = ''; // removes pb from parentNode before 10ms timer fires
    await new Promise((r) => setTimeout(r, 40));

    document.body.removeChild(container);
    openSpy.mockRestore();
  });
});
