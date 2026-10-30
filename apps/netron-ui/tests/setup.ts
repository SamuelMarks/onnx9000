const g = globalThis as typeof globalThis & {
  self: unknown;
  postMessage: () => void;
  workerInstances: unknown[];
  Worker: unknown;
  Blob: unknown;
  File: unknown;
};

g.self = globalThis;
g.postMessage = () => undefined;

Object.defineProperty(globalThis, 'navigator', {
  value: { userAgent: 'node.js' },
  writable: true,
});

const mockContext = {
  fillRect: () => undefined,
  clearRect: () => undefined,
  getImageData: () => ({
    data: new Uint8ClampedArray(4),
  }),
  putImageData: () => undefined,
  createImageData: () => [],
  setTransform: () => undefined,
  drawImage: () => undefined,
  save: () => undefined,
  restore: () => undefined,
  scale: () => undefined,
  translate: () => undefined,
  rotate: () => undefined,
  transform: () => undefined,
  beginPath: () => undefined,
  closePath: () => undefined,
  arc: () => undefined,
  arcTo: () => undefined,
  moveTo: () => undefined,
  lineTo: () => undefined,
  quadraticCurveTo: () => undefined,
  bezierCurveTo: () => undefined,
  rect: () => undefined,
  roundRect: () => undefined,
  fill: () => undefined,
  stroke: () => undefined,
  clip: () => undefined,
  fillText: () => undefined,
  strokeText: () => undefined,
  measureText: () => ({ width: 0 }),
  isPointInPath: () => false,
  isPointInStroke: () => false,
  createLinearGradient: () => ({
    addColorStop: () => undefined,
  }),
  createRadialGradient: () => ({
    addColorStop: () => undefined,
  }),
  createPattern: () => ({}),
};

if (typeof HTMLCanvasElement !== 'undefined') {
  (HTMLCanvasElement.prototype as { getContext: unknown }).getContext = () => mockContext;
}

g.workerInstances = [];
class MockWorker {
  constructor() {
    g.workerInstances.push(this);
  }
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;
  postMessage(_data: unknown) {
    if (this.onmessage) {
      this.onmessage({
        data: {
          type: 'PARSE_SUCCESS',
          layout: {
            nodes: [
              {
                id: '1',
                name: 'AddNode',
                opType: 'Add',
                type: 'node',
              },
              { id: 'input_X', name: 'X', type: 'input' },
              {
                id: 'output_Y',
                name: 'Y',
                type: 'output',
              },
              {
                id: 'constant_W',
                name: 'W',
                type: 'constant',
              },
            ],
            edges: [],
          },
          graph: {
            nodes: [
              {
                name: 'AddNode',
                opType: 'Add',
                inputs: ['X', 'W'],
                outputs: ['Y'],
                attributes: {
                  attr1: { type: 'FLOAT', value: 1.0 },
                },
                domain: '',
              },
            ],
            tensors: {
              W: { name: 'W', dtype: 'float32', shape: [1], size: 1 },
            },
            inputs: [{ name: 'X', dtype: 'float32', shape: [1] }],
            outputs: [{ name: 'Y', dtype: 'float32', shape: [1] }],
            initializers: ['W'],
          },
        },
      });
    }
  }
  terminate() {}
}
g.Worker = MockWorker;

if (typeof Blob === 'undefined') {
  g.Blob = class Blob {
    constructor(public parts: unknown[]) {}
  };
}
if (typeof File === 'undefined') {
  g.File = class File extends Blob {
    constructor(
      parts: BlobPart[],
      public name: string,
    ) {
      super(parts);
    }
  };
}

global.Path2D = class Path2D {
  moveTo() {}
  lineTo() {}
  arc() {}
  closePath() {}
  bezierCurveTo() {}
  rect() {}
} as typeof Path2D;
