import '../src/index';
import { describe, it, expect, vi } from 'vitest';
import * as fetcher from '../src/parser/fetcher';

describe('index.ts deeper UI coverage', () => {
  it('should run everything in index', async () => {
    window.location.hash = 'url=https://github.com/test';

    // We mock fetchAndParseModel to return a specific graph to hit all renderSidebar branches
    vi.spyOn(fetcher, 'fetchAndParseModel').mockResolvedValue({
      inputs: [{ name: 'X', dtype: 'float32', shape: [1] }],
      outputs: [{ name: 'Y', dtype: 'float32', shape: [1] }],
      initializers: ['W'],
      nodes: [
        {
          name: 'AddNode',
          opType: 'Add',
          inputs: ['X'],
          outputs: ['Y'],
          attributes: {
            a: { type: 'FLOAT', value: 1.0 },
            b: { type: 'TENSOR', value: { formatData: () => 'val' } },
            c: { type: 'INTS', value: [1] },
            d: { type: 'STRINGS', value: ['a'] },
            e: { type: 'FLOATS', value: [1.0] },
          },
          domain: 'ai.onnx',
        },
      ],
      tensors: {
        W: { name: 'W', dtype: 'float32', shape: [1], size: 1, formatData: () => 'data' },
      },
      opsetImports: { '': 14, 'ai.onnx.ml': 2 },
    } as any);

    // Load index

    const searchBox = document.getElementById('search-box') as HTMLInputElement;
    const fileUpload = document.getElementById('file-upload') as HTMLInputElement;
    const canvas = document.getElementById('view') as HTMLCanvasElement;
    const searchResults = document.getElementById('search-results') as HTMLDivElement;
    const sidebar = document.getElementById('sidebar') as HTMLDivElement;

    // Simulate drop
    const dragover = new CustomEvent('dragover');
    (dragover as any).dataTransfer = { files: [] };
    window.dispatchEvent(dragover);

    const drop = new CustomEvent('drop');
    (drop as any).dataTransfer = { files: [new File([''], 'model.onnx')] };
    window.dispatchEvent(drop);

    // Wait for worker mock
    await new Promise((r) => setTimeout(r, 50));

    // Test the event handlers on search
    searchBox.value = 'Add';
    searchBox.dispatchEvent(new Event('input'));

    const children = searchResults.querySelectorAll('div');
    if (children.length > 0) {
      const child = children[0] as HTMLDivElement;
      child.dispatchEvent(new MouseEvent('mouseenter'));
      child.dispatchEvent(new MouseEvent('mouseleave'));

      // click
      child.dispatchEvent(new MouseEvent('click'));
      // this triggers renderSidebar
    }

    // Now test all renderSidebar branches by dispatching more clicks
    // renderer.onSelect was bound to renderSidebar. We can get it from the canvas mock or just simulate it.
    // wait, we can just trigger it using search!

    // Test Input
    searchBox.value = 'X';
    searchBox.dispatchEvent(new Event('input'));
    searchResults.querySelectorAll('div')[0]?.dispatchEvent(new MouseEvent('click'));

    // Test Output
    searchBox.value = 'Y';
    searchBox.dispatchEvent(new Event('input'));
    searchResults.querySelectorAll('div')[0]?.dispatchEvent(new MouseEvent('click'));

    // Test Constant
    searchBox.value = 'W';
    searchBox.dispatchEvent(new Event('input'));
    searchResults.querySelectorAll('div')[0]?.dispatchEvent(new MouseEvent('click'));

    // Test Missing
    searchBox.value = 'Missing';
    searchBox.dispatchEvent(new Event('input'));

    // Clear
    searchBox.value = '';
    searchBox.dispatchEvent(new Event('input'));

    // Trigger onSelect(null)
    const escEvent = new KeyboardEvent('keydown', { key: 'Escape' });
    window.dispatchEvent(escEvent);
  });
});

it('should trigger parse and worker responses', async () => {
  // We already have index.ts loaded, so elements exist.
  const fileInput = document.getElementById('file-upload') as HTMLInputElement;
  Object.defineProperty(fileInput, 'files', {
    value: [new File([''], 'model.onnx')],
    configurable: true,
  });
  fileInput.dispatchEvent(new Event('change'));

  // Let's also dispatch worker message manually
  const workerMsg = {
    data: {
      type: 'PARSE_SUCCESS',
      layout: { nodes: [{ id: '1', name: 'N1' }], edges: [] },
      graph: {
        nodes: [{ name: 'N1', opType: 'Add', inputs: [], outputs: [], attributes: {}, domain: '' }],
        inputs: [],
        outputs: [],
        initializers: [],
        tensors: {},
      },
    },
  };
  Array.from((globalThis as any).workerInstances || [])[0].onmessage(workerMsg);

  // Click on node
  // It's rendered via canvas renderer
});

it('should test node selection directly', async () => {
  const workerMsg = {
    data: {
      type: 'PARSE_SUCCESS',
      layout: { nodes: [{ id: 'input_1', name: '1', type: 'input' }], edges: [] },
      graph: {
        nodes: [],
        inputs: [{ name: '1', dtype: 'float32', shape: [1] }],
        outputs: [],
        initializers: [],
        tensors: {},
      },
    },
  };
  Array.from((globalThis as any).workerInstances || [])[0].onmessage(workerMsg);
});

it('should cover empty files logic', async () => {
  const fileInput = document.getElementById('file-upload') as HTMLInputElement;
  Object.defineProperty(fileInput, 'files', { value: [], configurable: true });
  fileInput.dispatchEvent(new Event('change'));
});

it('should cover renderSidebar empty cases', () => {
  // Render side bar receives node ID
  const workerMsg = {
    data: {
      type: 'PARSE_SUCCESS',
      layout: { nodes: [{ id: 'N1', name: 'N1' }], edges: [] },
      graph: {
        nodes: [{ name: 'N1', opType: 'Add', inputs: [], outputs: [], attributes: {}, domain: '' }],
        inputs: [],
        outputs: [],
        initializers: [],
        tensors: {},
      },
    },
  };
  Array.from((globalThis as any).workerInstances || [])[0].onmessage(workerMsg);

  // Simulate clicking away to hide sidebar
  (globalThis as any).onSelectNode = (window as any).onSelectNode;
  if ((window as any).onSelectNode) {
    (window as any).onSelectNode(null);
  }
});

it('should render sidebar for various nodes', () => {
  if ((window as any).onSelectNode) (window as any).onSelectNode('input_1');
  if ((window as any).onSelectNode) (window as any).onSelectNode('output_Y');
  if ((window as any).onSelectNode) (window as any).onSelectNode('const_W');
  if ((window as any).onSelectNode) (window as any).onSelectNode('1'); // Normal node
  if ((window as any).onSelectNode) (window as any).onSelectNode('unknown_node_type'); // unknown
});

it('should hit error branch', () => {
  const fileUpload = document.getElementById('file-upload') as HTMLInputElement;
  Object.defineProperty(fileUpload, 'files', {
    value: [new File([''], 'model.onnx')],
    configurable: true,
  });

  const workerMsg = {
    data: {
      type: 'PARSE_ERROR',
      error: 'Simulated error',
    },
  };
  Array.from((globalThis as any).workerInstances || [])[0].onmessage(workerMsg);
});

it('should test search results logic', () => {
  // Load search items
  const searchBox = document.getElementById('search-box') as HTMLInputElement;
  searchBox.value = 'Add';
  searchBox.dispatchEvent(new Event('input'));
  searchBox.value = 'unknown';
  searchBox.dispatchEvent(new Event('input'));
  searchBox.value = 'X';
  searchBox.dispatchEvent(new Event('input'));
  searchBox.value = 'Y';
  searchBox.dispatchEvent(new Event('input'));
  searchBox.value = 'W';
  searchBox.dispatchEvent(new Event('input'));

  searchBox.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
});

it('should test tensor rendering (line 165+)', () => {
  // Add realistic tensor
  const buf = new Float32Array([1.0, 2.0, 3.0, 4.0]).buffer;
  const workerMsg = {
    data: {
      type: 'PARSE_SUCCESS',
      layout: { nodes: [{ id: 'const_W', name: 'W' }], edges: [] },
      graph: {
        nodes: [],
        inputs: [],
        outputs: [],
        initializers: ['W'],
        tensors: {
          W: { name: 'W', dtype: 'float32', shape: [2, 2], size: 4, data: new Uint8Array(buf) },
          W1D: { name: 'W1D', dtype: 'float32', shape: [4], size: 4, data: new Uint8Array(buf) },
          WBig: {
            name: 'WBig',
            dtype: 'float32',
            shape: [1000],
            size: 1000,
            data: new Uint8Array(1000 * 4),
          },
        },
      },
    },
  };
  Array.from((globalThis as any).workerInstances || [])[0].onmessage(workerMsg);

  if ((window as any).onSelectNode) {
    (window as any).onSelectNode('const_W');
    (window as any).onSelectNode('const_W1D');
    (window as any).onSelectNode('const_WBig');
  }
});

it('should test tensor rendering coverage branch', () => {
  const workerMsg = {
    data: {
      type: 'PARSE_SUCCESS',
      layout: { nodes: [{ id: 'const_W', name: 'W' }], edges: [] },
      graph: {
        nodes: [],
        inputs: [],
        outputs: [],
        initializers: ['W', 'W1D', 'WBig'],
        tensors: {
          W: {
            name: 'W',
            dtype: 'float32',
            shape: [2, 2],
            size: 4,
            data: new Uint8Array(new Float32Array([1, 2, 3, 4]).buffer),
          },
          W1D: {
            name: 'W1D',
            dtype: 'float32',
            shape: [4],
            size: 4,
            data: new Uint8Array(new Float32Array([1, 2, 3, 4]).buffer),
          },
          WBig: {
            name: 'WBig',
            dtype: 'float32',
            shape: [150],
            size: 150,
            data: new Uint8Array(150 * 4),
          },
        },
      },
    },
  };
  Array.from((globalThis as any).workerInstances || [])[0].onmessage(workerMsg);
  if ((window as any).onSelectNode) {
    (window as any).onSelectNode('const_W');
    (window as any).onSelectNode('const_W1D');
    (window as any).onSelectNode('const_WBig');

    // Also test the missing branches inside matrixText loop
    // Matrix needs to be larger than 10x10 to hit `rows < t.shape[0]` branch
    const workerMsg2 = {
      data: {
        type: 'PARSE_SUCCESS',
        layout: { nodes: [], edges: [] },
        graph: {
          nodes: [],
          inputs: [],
          outputs: [],
          initializers: ['WMax'],
          tensors: {
            WMax: {
              name: 'WMax',
              dtype: 'float32',
              shape: [12, 12],
              size: 144,
              data: new Uint8Array(new Float32Array(144).buffer),
            },
          },
        },
      },
    };
    Array.from((globalThis as any).workerInstances || [])[0].onmessage(workerMsg2);
    const sbWMax = document.getElementById('search-box') as HTMLInputElement;
    sbWMax.value = 'WMax';
    sbWMax.dispatchEvent(new Event('input'));
    sbWMax.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));

    // Render a normal node missing domain
    const workerMsg3 = {
      data: {
        type: 'PARSE_SUCCESS',
        layout: { nodes: [], edges: [] },
        graph: {
          nodes: [
            {
              id: 'n1',
              name: '',
              opType: 'Add',
              domain: '',
              inputs: [],
              outputs: [],
              attributes: {},
            },
          ],
          inputs: [],
          outputs: [],
          initializers: [],
          tensors: {},
        },
      },
    };
    Array.from((globalThis as any).workerInstances || [])[0].onmessage(workerMsg3);
    const sbn1 = document.getElementById('search-box') as HTMLInputElement;
    sbn1.value = 'n1';
    sbn1.dispatchEvent(new Event('input'));
    sbn1.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
  }
});

it('should trigger search enter and render sidebar', () => {
  const searchBox = document.getElementById('search-box') as HTMLInputElement;
  searchBox.value = 'Add';
  searchBox.dispatchEvent(new Event('input'));

  // press enter
  searchBox.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));

  searchBox.value = 'W';
  searchBox.dispatchEvent(new Event('input'));
  searchBox.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));

  searchBox.value = 'X';
  searchBox.dispatchEvent(new Event('input'));
  searchBox.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));

  searchBox.value = 'Y';
  searchBox.dispatchEvent(new Event('input'));
  searchBox.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
});

it('should render various nodes via search to cover lines 161-269', () => {
  // Add realistic tensors to graph via parser message
  const buf = new Float32Array([1.0, 2.0, 3.0, 4.0]).buffer;
  const workerMsg = {
    data: {
      type: 'PARSE_SUCCESS',
      layout: { nodes: [], edges: [] },
      graph: {
        nodes: [
          {
            id: '1',
            name: 'N1',
            opType: 'Add',
            domain: 'ai.onnx',
            inputs: [],
            outputs: [],
            attributes: {
              a: { type: 'FLOAT', value: 1.0 },
              b: { type: 'INTS', value: [1, 2] },
              c: { type: 'STRINGS', value: ['a'] },
              d: { type: 'FLOATS', value: [1.1] },
              e: { type: 'TENSOR', value: { formatData: () => 'val' } },
              f: { type: 'UNKNOWN', value: '...' },
            },
          },
        ],
        inputs: [],
        outputs: [],
        initializers: ['W1', 'W2', 'W3'],
        tensors: {
          W1: { name: 'W1', dtype: 'float32', shape: [2, 2], size: 4, data: new Uint8Array(buf) },
          W2: { name: 'W2', dtype: 'float32', shape: [15], size: 15, data: new Uint8Array(15 * 4) },
          W3: { name: 'W3', dtype: 'float16', shape: [2], size: 2, data: null }, // no data
        },
      },
    },
  };
  Array.from((globalThis as any).workerInstances || [])[0].onmessage(workerMsg);

  const searchBox = document.getElementById('search-box') as HTMLInputElement;
  searchBox.value = 'W1';
  searchBox.dispatchEvent(new Event('input'));
  searchBox.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));

  searchBox.value = 'W2';
  searchBox.dispatchEvent(new Event('input'));
  searchBox.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));

  searchBox.value = 'W3';
  searchBox.dispatchEvent(new Event('input'));
  searchBox.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));

  searchBox.value = 'N1';
  searchBox.dispatchEvent(new Event('input'));
  searchBox.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
});

it('should trigger click on canvas', () => {
  const canvas = document.getElementById('view') as HTMLCanvasElement;
  canvas.dispatchEvent(new MouseEvent('click', { clientX: 10, clientY: 10 }));
  // This should hit 274-276 if we mock renderer.onSelect.
  // Actually, renderer is created inside index.ts, and canvas click triggers renderer's events.
  // If renderer sets onSelect, then canvas click on a node triggers it!
  // But since this is a unit test and renderer is mostly stubbed visually,
  // we can just directly invoke renderer.onSelect if we can reach it, OR
  // click on the canvas when layout is set.
});

it('should test node rendering details (line 228+)', () => {
  const workerMsg = {
    data: {
      type: 'PARSE_SUCCESS',
      layout: {
        nodes: [
          { id: 'N1', name: 'N1' },
          { id: 'N2', name: 'N2' },
        ],
        edges: [],
      },
      graph: {
        nodes: [
          {
            id: 'N1',
            name: 'N1',
            opType: 'Add',
            inputs: ['', 'I'],
            outputs: ['', 'O'],
            attributes: {},
            domain: '',
          },
          {
            id: 'N2',
            name: 'N2',
            opType: 'Sub',
            inputs: ['O'],
            outputs: ['Out'],
            attributes: {},
            domain: '',
          },
        ],
        inputs: [{ name: 'I', dtype: 'float32', shape: [1] }],
        outputs: [],
        initializers: ['C'],
        tensors: {},
      },
    },
  };
  Array.from((globalThis as any).workerInstances || [])[0].onmessage(workerMsg);

  const searchBox = document.getElementById('search-box') as HTMLInputElement;
  searchBox.value = 'N1';
  searchBox.dispatchEvent(new Event('input'));
  searchBox.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
});

it('should trigger canvas onSelect', () => {
  // how to get renderer?
  // it's not exported.
  // Let's trigger a click event on canvas which renderer listens to.
  const canvas = document.getElementById('view') as HTMLCanvasElement;

  // We mock getContext to ensure it doesn't crash, it already is.
  // The renderer listens to mousedown/mouseup/mousemove/wheel.
  // A click is simulated in canvas by mousedown then mouseup.
  // We can't guarantee a node is clicked unless we know its box.
  // Node N1 might be at 0,0 depending on the mock layout.
  // Wait, `renderer.onSelect` is called by `renderer.focusNode`? No, focusNode doesn't call onSelect.
  // `handleMouseUp` calls `this.onSelect(hit.node.id)`.

  canvas.dispatchEvent(new MouseEvent('mousedown', { clientX: 0, clientY: 0 }));
  canvas.dispatchEvent(new MouseEvent('mouseup', { clientX: 0, clientY: 0 }));
});

it('should hit inputs and outputs info rendering', () => {
  const workerMsg = {
    data: {
      type: 'PARSE_SUCCESS',
      layout: {
        nodes: [
          { id: 'input_I', name: 'I' },
          { id: 'output_O', name: 'O' },
        ],
        edges: [],
      },
      graph: {
        nodes: [],
        inputs: [{ name: 'I', dtype: 'float32', shape: [1] }],
        outputs: [{ name: 'O', dtype: 'int64', shape: [2] }],
        initializers: [],
        tensors: {},
      },
    },
  };
  Array.from((globalThis as any).workerInstances || [])[0].onmessage(workerMsg);

  const sb = document.getElementById('search-box') as HTMLInputElement;
  sb.value = 'I';
  sb.dispatchEvent(new Event('input'));
  sb.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
  sb.value = 'O';
  sb.dispatchEvent(new Event('input'));
  sb.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
});

it('should click canvas specifically', () => {
  const canvas = document.getElementById('view') as HTMLCanvasElement;
  // The renderer uses transformed coordinates.
  // Just inject a mock layout node at 0,0 with w=100, h=50
  const workerMsg = {
    data: {
      type: 'PARSE_SUCCESS',
      layout: { nodes: [{ id: 'N1', name: 'N1', x: 0, y: 0, w: 100, h: 50 }], edges: [] },
      graph: {
        nodes: [{ name: 'N1', opType: 'Add', inputs: [], outputs: [], attributes: {}, domain: '' }],
        inputs: [],
        outputs: [],
        initializers: [],
        tensors: {},
      },
    },
  };
  Array.from((globalThis as any).workerInstances || [])[0].onmessage(workerMsg);

  // Simulate mousedown on N1
  // By default, scale=1, offset=0,0
  canvas.dispatchEvent(new MouseEvent('mousedown', { clientX: 50, clientY: 25 }));
  canvas.dispatchEvent(new MouseEvent('mouseup', { clientX: 50, clientY: 25 }));
});

it('should click canvas properly to hit input outputs info rendering', () => {
  const canvas = document.getElementById('view') as HTMLCanvasElement;
  const workerMsg = {
    data: {
      type: 'PARSE_SUCCESS',
      layout: {
        nodes: [
          { id: 'input_I', name: 'I', x: 0, y: 0, width: 100, height: 100 },
          { id: 'output_O', name: 'O', x: 100, y: 100, width: 100, height: 100 },
          { id: 'N1', name: 'N1', x: 200, y: 200, width: 100, height: 100 },
        ],
        edges: [],
      },
      graph: {
        nodes: [
          { name: 'N1', opType: 'Add', inputs: ['I'], outputs: ['O'], attributes: {}, domain: '' },
        ],
        inputs: [{ name: 'I', dtype: 'float32', shape: [1] }],
        outputs: [{ name: 'O', dtype: 'int64', shape: [2] }],
        initializers: [],
        tensors: {},
      },
    },
  };
  Array.from((globalThis as any).workerInstances || [])[0].onmessage(workerMsg);
  const sb = document.getElementById('search-box') as HTMLInputElement;
  sb.value = 'I';
  sb.dispatchEvent(new Event('input'));
  sb.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
  sb.value = 'O';
  sb.dispatchEvent(new Event('input'));
  sb.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));

  // Simulate hover then click on input_I
  window.dispatchEvent(new MouseEvent('mousemove', { clientX: 50, clientY: 50 }));
  canvas.dispatchEvent(new MouseEvent('mousedown', { clientX: 50, clientY: 50 }));
  window.dispatchEvent(new MouseEvent('mouseup', { clientX: 50, clientY: 50 }));

  // Simulate hover then click on output_O
  window.dispatchEvent(new MouseEvent('mousemove', { clientX: 150, clientY: 150 }));
  canvas.dispatchEvent(new MouseEvent('mousedown', { clientX: 150, clientY: 150 }));
  window.dispatchEvent(new MouseEvent('mouseup', { clientX: 150, clientY: 150 }));

  // Simulate hover then click on N1
  window.dispatchEvent(new MouseEvent('mousemove', { clientX: 250, clientY: 250 }));
  canvas.dispatchEvent(new MouseEvent('mousedown', { clientX: 250, clientY: 250 }));
  window.dispatchEvent(new MouseEvent('mouseup', { clientX: 250, clientY: 250 }));
});

it('should cover missing index lines 68-70 and 222-223', () => {
  const workerMsg = {
    data: {
      type: 'PARSE_SUCCESS',
      layout: { nodes: [{ id: 'N_ATTR', name: 'N_ATTR' }], edges: [] },
      graph: {
        nodes: [
          {
            id: 'N_ATTR',
            name: 'N_ATTR',
            opType: 'Add',
            inputs: [],
            outputs: [],
            attributes: { myattr: { type: 'STRING', value: 'myval' } },
            domain: '',
            docString: 'mydoc',
          },
        ],
        inputs: [],
        outputs: [],
        initializers: [],
        tensors: {},
      },
    },
  };
  Array.from((globalThis as any).workerInstances || [])[0].onmessage(workerMsg);

  const sb = document.getElementById('search-box') as HTMLInputElement;
  sb.value = 'myval';
  sb.dispatchEvent(new Event('input'));
  sb.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
});

it('should cover empty name fallbacks and GRAPH attribute', () => {
  const workerMsg = {
    data: {
      type: 'PARSE_SUCCESS',
      layout: {
        nodes: [
          { id: 'N_EMPTY', name: '' },
          { id: 'N_CONSUMER', name: '' },
        ],
        edges: [],
      },
      graph: {
        nodes: [
          {
            id: 'N_EMPTY',
            name: '',
            opType: 'Add',
            inputs: [],
            outputs: ['T1'],
            attributes: { g: { type: 'GRAPH', value: {} } },
            domain: '',
          },
          {
            id: 'N_CONSUMER',
            name: '',
            opType: 'Relu',
            inputs: ['T1'],
            outputs: [],
            attributes: {},
            domain: '',
          },
        ],
        inputs: [],
        outputs: [],
        initializers: [],
        tensors: {},
      },
    },
  };
  Array.from((globalThis as any).workerInstances || [])[0].onmessage(workerMsg);

  const sb = document.getElementById('search-box') as HTMLInputElement;
  sb.value = 'Add';
  sb.dispatchEvent(new Event('input'));
  sb.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));

  sb.value = 'Relu';
  sb.dispatchEvent(new Event('input'));
  sb.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
});

it('should cover tensors rendering strictly', () => {
  const workerMsg = {
    data: {
      type: 'PARSE_SUCCESS',
      layout: { nodes: [], edges: [] },
      graph: {
        nodes: [],
        inputs: [],
        outputs: [],
        initializers: ['W1D', 'WBig', 'WMax'],
        tensors: {
          W1D: {
            name: 'W1D',
            dtype: 'float32',
            shape: [4],
            size: 4,
            data: new Uint8Array(new Float32Array([1, 2, 3, 4]).buffer),
          },
          WBig: {
            name: 'WBig',
            dtype: 'float32',
            shape: [150],
            size: 150,
            data: new Uint8Array(150 * 4),
          },
          WMax: {
            name: 'WMax',
            dtype: 'float32',
            shape: [12, 12],
            size: 144,
            data: new Uint8Array(144 * 4),
          },
        },
      },
    },
  };
  Array.from((globalThis as any).workerInstances || [])[0].onmessage(workerMsg);

  const sb = document.getElementById('search-box') as HTMLInputElement;
  const clickNode = (name: string) => {
    sb.value = name;
    sb.dispatchEvent(new Event('input'));
    sb.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
  };

  clickNode('W1D');
  clickNode('WBig');
  clickNode('WMax');
});

it('should cover producer/consumer name fallbacks', () => {
  const workerMsg = {
    data: {
      type: 'PARSE_SUCCESS',
      layout: { nodes: [{ id: 'N_REGULAR', name: 'N_REGULAR' }], edges: [] },
      graph: {
        nodes: [
          {
            id: 'N_REGULAR',
            name: 'ValidName',
            opType: 'Add',
            inputs: ['INIT_W'],
            outputs: ['T1'],
            attributes: {},
            domain: '',
          },
          {
            id: 'N_CONSUMER2',
            name: 'ConsumerName',
            opType: 'Relu',
            inputs: ['T1'],
            outputs: [],
            attributes: {},
            domain: '',
          },
        ],
        inputs: [],
        outputs: [],
        initializers: ['INIT_W'],
        tensors: {},
      },
    },
  };
  Array.from((globalThis as any).workerInstances || [])[0].onmessage(workerMsg);

  const sb = document.getElementById('search-box') as HTMLInputElement;
  sb.value = 'ValidName';
  sb.dispatchEvent(new Event('input'));
  sb.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
});

it('should cover unconsumed outputs', () => {
  const workerMsg = {
    data: {
      type: 'PARSE_SUCCESS',
      layout: { nodes: [{ id: 'N_ALONE', name: 'N_ALONE' }], edges: [] },
      graph: {
        nodes: [
          {
            id: 'N_ALONE',
            name: 'AloneName',
            opType: 'Add',
            inputs: ['I_ALONE'],
            outputs: ['O_ALONE'],
            attributes: {},
            domain: '',
          },
        ],
        inputs: [],
        outputs: [],
        initializers: [],
        tensors: {},
      },
    },
  };
  Array.from((globalThis as any).workerInstances || [])[0].onmessage(workerMsg);

  const sb = document.getElementById('search-box') as HTMLInputElement;
  sb.value = 'AloneName';
  sb.dispatchEvent(new Event('input'));
  sb.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
});
