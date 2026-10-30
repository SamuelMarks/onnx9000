import { beforeEach, describe, expect, it, vi } from 'vitest';

let lastWorker: {
  postMessage: (msg: unknown) => void;
  onmessage: ((e: { data: unknown }) => void) | null;
} | null = null;

let mockExportShouldFail: false | 'error' | 'string' = false;

vi.mock('@onnx9000/modifier/dist/GraphMutator.js', () => ({
  GraphMutator: class {
    constructor(public graph: Record<string, unknown>) {}
    removeNode(id: string) {
      const nodes = this.graph.nodes as { id: string }[];
      this.graph.nodes = nodes.filter((n) => n.id !== id);
    }
    addNode(
      opType: string,
      inputs: string[],
      outputs: string[],
      attributes: Record<string, unknown>,
      name: string,
    ) {
      (this.graph.nodes as unknown[]).push({
        id: name,
        opType,
        inputs,
        outputs,
        attributes,
        name,
      });
    }
    setNodeAttribute(name: string, k: string, value: unknown, type: string) {
      const nodes = this.graph.nodes as {
        id: string;
        name: string;
        attributes: Record<string, unknown>;
      }[];
      const n = nodes.find((x) => x.name === name || x.id === name);
      if (n) n.attributes[k] = { value, type };
    }
    extractSubgraph() {
      if (mockExportShouldFail === 'error') throw new Error('subgraph error');
      if (mockExportShouldFail === 'string') throw 'subgraph string error';
      return this.graph;
    }
  },
}));

vi.mock('@onnx9000/modifier/dist/components/export/exporter.js', () => ({
  ModelExporter: class {
    async exportModel() {
      return new Uint8Array([1, 2, 3]);
    }
    downloadBlob() {}
    generatePythonHelperScript() {
      return 'import onnx9000';
    }
  },
}));

vi.mock('@onnx9000/modifier/dist/components/utilities.js', () => ({
  ModifierUtilities: class {
    autoFormatNodeNames() {}
  },
}));

vi.mock('../src/render/canvas', () => ({
  CanvasRenderer: class {
    layout: unknown = null;
    selectedNodes: string[] = [];
    onSelect: (nodeId: string | null) => void = () => {};
    setLayout(l: unknown) {
      this.layout = l;
    }
    setFilterControlEdges() {}
    setCustomColorRegex() {}
    setSearchResults() {}
    focusNode() {}
    render() {}
  },
}));

describe('index.ts', () => {
  beforeEach(() => {
    document.body.innerHTML = '';
    vi.resetModules();
    lastWorker = null;
    mockExportShouldFail = false;
    const workerMock = {
      onmessage: null as ((e: { data: unknown }) => void) | null,
      postMessage: () => {},
    };
    lastWorker = workerMock;
    (global as typeof globalThis & { Worker: unknown }).Worker = function Worker() {
      return workerMock;
    };
    window.alert = vi.fn();
    Object.assign(navigator, {
      clipboard: { writeText: vi.fn().mockResolvedValue(undefined) },
    });
    if (typeof HTMLCanvasElement !== 'undefined') {
      HTMLCanvasElement.prototype.toDataURL = vi.fn().mockReturnValue('data:image/png;base64,00');
      HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue({
        fillStyle: '',
        fillRect: vi.fn(),
      });
    }
  });

  it('should initialize UI and cover all interactive features', async () => {
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    await import('../src/index.ts');

    const canvas = document.getElementById('view') as HTMLCanvasElement;
    const dropZone = document.getElementById('drop-zone') as HTMLDivElement;
    const fileUpload = document.getElementById('file-upload') as HTMLInputElement;
    const filterControlEdges = document.getElementById('filter-control-edges') as HTMLInputElement;
    const colorRegex = document.getElementById('color-regex') as HTMLInputElement;
    const searchBox = document.getElementById('search-box') as HTMLInputElement;
    const btnHelp = document.getElementById('btn-help') as HTMLButtonElement;
    const closeHelpBtn = document.getElementById('close-help-btn') as HTMLButtonElement;
    const breadcrumb = document.getElementById('breadcrumb') as HTMLDivElement;

    const menuDuplicateNode = document.getElementById('menu-duplicate-node') as HTMLDivElement;
    const menuExportPng = document.getElementById('menu-export-png') as HTMLDivElement;
    const menuExportJson = document.getElementById('menu-export-json') as HTMLDivElement;
    const menuCopyAttributes = document.getElementById('menu-copy-attributes') as HTMLDivElement;
    const menuPasteAttributes = document.getElementById('menu-paste-attributes') as HTMLDivElement;
    const menuExtractSubgraph = document.getElementById('menu-extract-subgraph') as HTMLDivElement;
    const menuExtractPython = document.getElementById('menu-extract-python') as HTMLDivElement;
    const menuReplaceConstant = document.getElementById('menu-replace-constant') as HTMLDivElement;

    const modObj = (
      window as unknown as {
        onnxModifier: {
          getGraph: () => unknown;
          getRenderer: () => {
            selectedNodes: string[];
            layout: unknown;
            onSelect: (id: string | null) => void;
          };
          getMutator: () => unknown;
          download: () => Promise<void>;
        };
      }
    ).onnxModifier;

    expect(modObj.getGraph()).toBeNull();
    const renderer = modObj.getRenderer();
    expect(modObj.getMutator()).toBeDefined();
    await modObj.download(); // when currentGraph is null

    // Trigger menu items when currentGraph is null
    renderer.onSelect(null);
    menuDuplicateNode.click();
    menuExportPng.click();
    menuExportJson.click();
    menuCopyAttributes.click();
    menuPasteAttributes.click();
    menuExtractSubgraph.click();
    menuExtractPython.click();
    menuReplaceConstant.click();
    breadcrumb.click();

    // Help modal & view options
    btnHelp.click();
    expect(document.getElementById('help-modal')?.style.display).toBe('block');
    closeHelpBtn.click();
    expect(document.getElementById('help-modal')?.style.display).toBe('none');

    filterControlEdges.checked = true;
    filterControlEdges.dispatchEvent(new Event('change'));
    colorRegex.value = '^Conv';
    colorRegex.dispatchEvent(new Event('input'));

    // Drag, drop, click, fileUpload change
    const clickSpy = vi.spyOn(fileUpload, 'click').mockImplementation(() => {});
    dropZone.click();
    expect(clickSpy).toHaveBeenCalled();

    window.dispatchEvent(new Event('dragover'));
    window.dispatchEvent(new Event('dragleave'));
    window.dispatchEvent(Object.assign(new Event('drop'), { dataTransfer: { files: [] } }));
    const mockFile = new File(['mock content'], 'model.onnx', {
      type: 'application/octet-stream',
    });
    window.dispatchEvent(Object.assign(new Event('drop'), { dataTransfer: { files: [mockFile] } }));
    Object.defineProperty(fileUpload, 'files', { value: [], configurable: true });
    fileUpload.dispatchEvent(new Event('change'));
    Object.defineProperty(fileUpload, 'files', { value: [mockFile], configurable: true });
    fileUpload.dispatchEvent(new Event('change'));

    // Worker PARSE_ERROR and PARSE_SUCCESS
    lastWorker?.onmessage?.({ data: { type: 'PARSE_ERROR', error: 'bad parse' } });
    expect(document.getElementById('status')?.textContent).toContain('Error: bad parse');

    // Construct rich graph with 2D float32 weight (>10 rows/cols), 1D float32 weight (>100 items), Cast to double (11), float64 input, docString, optional inputs/outputs, TENSOR/GRAPH attributes
    const f32Data2D = new Float32Array(12 * 12);
    for (let i = 0; i < f32Data2D.length; i++) f32Data2D[i] = i - 50;
    const f32Data1D = new Float32Array(105);
    for (let i = 0; i < f32Data1D.length; i++) f32Data1D[i] = i * 0.1;

    const richGraph = {
      name: '',
      producerName: 'prod',
      producerVersion: '1.0',
      domain: '',
      modelVersion: 1,
      opsetImports: { 'ai.onnx': 18 },
      inputs: [
        { name: 'in_f64', dtype: 'float64', shape: [1, 4] },
        { name: 'in_f32', dtype: 'float32', shape: [1, 4] },
      ],
      outputs: [{ name: 'out_y', dtype: 'float32', shape: [1, 4] }],
      initializers: ['w_2d', 'w_1d', 'w_small_2d', 'w_small_1d'],
      tensors: {
        w_2d: {
          name: 'w_2d',
          dtype: 'float32',
          shape: [12, 12],
          size: 144,
          data: new Uint8Array(f32Data2D.buffer),
        },
        w_1d: {
          name: 'w_1d',
          dtype: 'float32',
          shape: [105],
          size: 105,
          data: new Uint8Array(f32Data1D.buffer),
        },
        w_small_2d: {
          name: 'w_small_2d',
          dtype: 'float32',
          shape: [2, 2],
          size: 4,
          data: new Uint8Array(new Float32Array([1, 2, 3, 4]).buffer),
        },
        w_small_1d: {
          name: 'w_small_1d',
          dtype: 'float32',
          shape: [2],
          size: 2,
          data: new Uint8Array(new Float32Array([1, 2]).buffer),
        },
      },
      nodes: [
        {
          id: 'n1',
          opType: 'Cast',
          name: 'cast_node',
          domain: '',
          docString: `**Bold** and *Italic*
Next line`,
          inputs: ['', 'in_f64', 'w_2d'],
          outputs: ['', 'mid_t'],
          attributes: {
            to: { value: 11, type: 'INT' },
            t_attr: { value: 't', type: 'TENSOR' },
            g_attr: { value: 'g', type: 'GRAPH' },
          },
        },
        {
          id: 'n2',
          opType: 'Cast',
          name: '',
          domain: 'custom.dom',
          docString: '',
          inputs: ['mid_t'],
          outputs: ['mid2_t'],
          attributes: {},
        },
        {
          id: 'n3',
          opType: 'Relu',
          name: '',
          domain: '',
          docString: '',
          inputs: ['mid2_t'],
          outputs: ['out_y'],
          attributes: {},
        },
      ],
    };

    const mockLayout = {
      width: 200,
      height: 200,
      nodes: [
        {
          id: 'n1',
          opType: 'Cast',
          name: 'cast_node',
          type: 'node',
          x: 0,
          y: 0,
          width: 100,
          height: 40,
        },
        { id: 'n2', opType: 'Cast', name: '', type: 'node', x: 0, y: 60, width: 100, height: 40 },
      ],
      edges: [],
    };

    lastWorker?.onmessage?.({
      data: { type: 'PARSE_SUCCESS', graph: richGraph, layout: mockLayout },
    });

    // Sidebar opset version input change & add opset button
    const verInp = document.querySelector('.opset-version-input') as HTMLInputElement;
    verInp.value = '19';
    verInp.dispatchEvent(new Event('change'));
    verInp.value = 'not_a_number';
    verInp.dispatchEvent(new Event('change'));

    const domInp = document.getElementById('new-opset-domain') as HTMLInputElement;
    const newVerInp = document.getElementById('new-opset-version') as HTMLInputElement;
    const addOpsetBtn = document.getElementById('add-opset-btn') as HTMLButtonElement;
    addOpsetBtn.click(); // empty domain
    domInp.value = 'com.microsoft';
    newVerInp.value = '1';
    addOpsetBtn.click();

    // Auto-format button
    const autoFmtBtn = document.getElementById('btn-auto-format') as HTMLButtonElement;
    autoFmtBtn.click();
    await new Promise((r) => setTimeout(r, 20));

    // Render sidebar for input_, output_, const_ (2D, 1D, small 2D, small 1D, and missing), normal nodes (n1, n2, n3, and missing)
    renderer.onSelect('input_in_f64');
    renderer.onSelect('input_missing');
    renderer.onSelect('output_out_y');
    renderer.onSelect('output_missing');
    renderer.onSelect('const_w_2d');
    renderer.onSelect('const_w_1d');
    renderer.onSelect('const_w_small_2d');
    renderer.onSelect('const_w_small_1d');
    renderer.onSelect('const_missing');
    renderer.onSelect('n1');
    renderer.onSelect('n2');
    renderer.onSelect('n3');
    renderer.onSelect('n_missing');

    // Render sidebar when opsetImports is undefined
    const savedOpsets = richGraph.opsetImports;
    (richGraph as { opsetImports?: unknown }).opsetImports = undefined;
    renderer.onSelect(null);
    richGraph.opsetImports = savedOpsets;

    // Search box input & Enter keydown
    searchBox.value = '   ';
    searchBox.dispatchEvent(new Event('input'));
    searchBox.value = 'cast_node';
    searchBox.dispatchEvent(new Event('input'));
    searchBox.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
    searchBox.value = 'mid_t';
    searchBox.dispatchEvent(new Event('input'));
    searchBox.value = 'to';
    searchBox.dispatchEvent(new Event('input'));
    searchBox.value = 'in_f64';
    searchBox.dispatchEvent(new Event('input'));
    searchBox.value = 'out_y';
    searchBox.dispatchEvent(new Event('input'));
    searchBox.value = 'w_2d';
    searchBox.dispatchEvent(new Event('input'));
    searchBox.value = 'non_existent_query_xyz';
    searchBox.dispatchEvent(new Event('input'));

    // Context menu with 0, 1 (node vs non-node), and 2 selected nodes; copy/paste attributes; duplicate; export png/json; extract subgraph/python; replace constant
    renderer.selectedNodes = [];
    canvas.dispatchEvent(new MouseEvent('contextmenu', { clientX: 10, clientY: 10 }));
    document.body.click();

    renderer.selectedNodes = ['input_in_f64'];
    canvas.dispatchEvent(new MouseEvent('contextmenu', { clientX: 10, clientY: 10 }));

    renderer.selectedNodes = ['n1'];
    menuCopyAttributes.click();
    menuDuplicateNode.click();
    canvas.dispatchEvent(new MouseEvent('contextmenu', { clientX: 10, clientY: 10 }));

    renderer.selectedNodes = ['n2'];
    menuPasteAttributes.click();
    menuExportPng.click();
    menuExportJson.click();
    menuDuplicateNode.click();
    await new Promise((r) => setTimeout(r, 20));

    renderer.selectedNodes = ['n1', 'n2'];
    canvas.dispatchEvent(new MouseEvent('contextmenu', { clientX: 10, clientY: 10 }));

    await menuExtractSubgraph.click();
    await menuExtractPython.click();
    menuReplaceConstant.click();

    mockExportShouldFail = 'error';
    await menuExtractSubgraph.click();
    await menuExtractPython.click();

    mockExportShouldFail = 'string';
    await menuExtractSubgraph.click();
    await menuExtractPython.click();
    mockExportShouldFail = false;

    // Keyboard navigation: Tab, Shift+Tab, Delete/Backspace, Escape, and Tab on empty layout
    searchBox.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', bubbles: true }));
    document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    renderer.selectedNodes = [];
    document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', bubbles: true }));
    document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', bubbles: true }));
    document.body.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'Tab', shiftKey: true, bubbles: true }),
    );
    document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'Delete', bubbles: true }));
    await new Promise((r) => setTimeout(r, 20));

    // Subgraph navigation (2 levels deep so breadcrumb tests both graphStack.length > 0 and === 0, plus Tab on empty layout)
    const subG1 = {
      ...richGraph,
      name: 'sub1',
      nodes: [],
      inputs: [],
      outputs: [],
      initializers: [],
      tensors: {},
    };
    const subG2 = {
      ...richGraph,
      name: 'sub2',
      nodes: [],
      inputs: [],
      outputs: [],
      initializers: [],
      tensors: {},
    };
    window.dispatchEvent(
      new CustomEvent('open-subgraph', { detail: { graph: subG1, name: 'sub1' } }),
    );
    window.dispatchEvent(
      new CustomEvent('open-subgraph', { detail: { graph: subG2, name: 'sub2' } }),
    );
    await new Promise((r) => setTimeout(r, 20));
    document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', bubbles: true }));
    breadcrumb.click();
    await new Promise((r) => setTimeout(r, 20));
    breadcrumb.click();
    await new Promise((r) => setTimeout(r, 20));

    await modObj.download();
    errSpy.mockRestore();
  });
});
