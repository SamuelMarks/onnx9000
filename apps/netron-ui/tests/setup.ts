(globalThis as Object).self = globalThis;
(globalThis as Object).postMessage = () => undefined;

Object.defineProperty(globalThis, 'navigator', { value: { userAgent: 'node.js' }, writable: true });

const mockContext = {
  fillRect: () => undefined,
  clearRect: () => undefined,
  getImageData: () => ({ data: new Uint8ClampedArray(4) }),
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
  createLinearGradient: () => ({ addColorStop: () => undefined }),
  createRadialGradient: () => ({ addColorStop: () => undefined }),
  createPattern: () => ({}),
};

if (typeof HTMLCanvasElement !== 'undefined') {
  (HTMLCanvasElement as Object).prototype.getContext = () => mockContext;
}

(globalThis as Object).workerInstances = [];
class MockWorker {
  constructor() {
    (globalThis as Object).workerInstances.push(this);
  }
  onmessage: Object;
  onerror: Object;
  postMessage(data: Object) {
    if (this.onmessage) {
      this.onmessage({
        data: {
          type: 'PARSE_SUCCESS',
          layout: {
            nodes: [
              { id: '1', name: 'AddNode', opType: 'Add', type: 'node' },
              { id: 'input_X', name: 'X', type: 'input' },
              { id: 'output_Y', name: 'Y', type: 'output' },
              { id: 'constant_W', name: 'W', type: 'constant' },
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
                attributes: { attr1: { type: 'FLOAT', value: 1.0 } },
                domain: '',
              },
            ],
            tensors: { W: { name: 'W', dtype: 'float32', shape: [1], size: 1 } },
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
(globalThis as Object).Worker = MockWorker;

if (typeof Blob === 'undefined') {
  (globalThis as Object).Blob = class Blob {
    constructor(public parts: Object[]) {}
  };
}
if (typeof File === 'undefined') {
  (globalThis as Object).File = class File extends Blob {
    constructor(
      parts: Object[],
      public name: string,
    ) {
      super(parts);
    }
  };
}

// @ts-ignore
global.Path2D = class Path2D {
  moveTo() {}
  lineTo() {}
  arc() {}
  closePath() {}
  bezierCurveTo() {}
  rect() {}
};
