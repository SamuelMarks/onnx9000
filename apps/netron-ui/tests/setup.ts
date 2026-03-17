(globalThis as any).self = globalThis;
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>');
(globalThis as any).window = dom.window;
(globalThis as any).document = dom.window.document;
(globalThis as any).navigator = { userAgent: 'node.js' };

const mockContext = {
  fillRect: () => {},
  clearRect: () => {},
  getImageData: () => ({ data: new Uint8ClampedArray(4) }),
  putImageData: () => {},
  createImageData: () => [],
  setTransform: () => {},
  drawImage: () => {},
  save: () => {},
  restore: () => {},
  scale: () => {},
  translate: () => {},
  rotate: () => {},
  transform: () => {},
  beginPath: () => {},
  closePath: () => {},
  arc: () => {},
  arcTo: () => {},
  moveTo: () => {},
  lineTo: () => {},
  quadraticCurveTo: () => {},
  bezierCurveTo: () => {},
  rect: () => {},
  roundRect: () => {},
  fill: () => {},
  stroke: () => {},
  clip: () => {},
  fillText: () => {},
  strokeText: () => {},
  measureText: () => ({ width: 0 }),
  isPointInPath: () => false,
  isPointInStroke: () => false,
  createLinearGradient: () => ({ addColorStop: () => {} }),
  createRadialGradient: () => ({ addColorStop: () => {} }),
  createPattern: () => ({}),
};

(globalThis as any).HTMLCanvasElement = dom.window.HTMLCanvasElement;
(globalThis as any).HTMLCanvasElement.prototype.getContext = () => mockContext;
dom.window.HTMLCanvasElement.prototype.getContext = () => mockContext;

class MockWorker {
  onmessage: any;
  onerror: any;
  postMessage(data: any) {
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
(globalThis as any).Worker = MockWorker;
dom.window.Worker = MockWorker;

if (typeof Blob === 'undefined') {
  (globalThis as any).Blob = class Blob {
    constructor(public parts: any[]) {}
  };
}
if (typeof File === 'undefined') {
  (globalThis as any).File = class File extends Blob {
    constructor(
      parts: any[],
      public name: string,
    ) {
      super(parts);
    }
  };
}
