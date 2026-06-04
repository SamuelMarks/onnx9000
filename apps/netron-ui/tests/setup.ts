/* v8 ignore next */ /* v8 ignore next */ (globalThis as Object).self =
  globalThis; /* v8 ignore next */ /* v8 ignore next */
(globalThis as Object).postMessage = () => undefined; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
Object.defineProperty(globalThis, 'navigator', {
  value: { userAgent: 'node.js' },
  writable: true,
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const mockContext = {
  /* v8 ignore next */ /* v8 ignore next */
  fillRect: () => undefined /* v8 ignore next */ /* v8 ignore next */,
  clearRect: () => undefined /* v8 ignore next */ /* v8 ignore next */,
  getImageData: () => ({
    data: new Uint8ClampedArray(4),
  }) /* v8 ignore next */ /* v8 ignore next */,
  putImageData: () => undefined /* v8 ignore next */ /* v8 ignore next */,
  createImageData: () => [] /* v8 ignore next */ /* v8 ignore next */,
  setTransform: () => undefined /* v8 ignore next */ /* v8 ignore next */,
  drawImage: () => undefined /* v8 ignore next */ /* v8 ignore next */,
  save: () => undefined /* v8 ignore next */ /* v8 ignore next */,
  restore: () => undefined /* v8 ignore next */ /* v8 ignore next */,
  scale: () => undefined /* v8 ignore next */ /* v8 ignore next */,
  translate: () => undefined /* v8 ignore next */ /* v8 ignore next */,
  rotate: () => undefined /* v8 ignore next */ /* v8 ignore next */,
  transform: () => undefined /* v8 ignore next */ /* v8 ignore next */,
  beginPath: () => undefined /* v8 ignore next */ /* v8 ignore next */,
  closePath: () => undefined /* v8 ignore next */ /* v8 ignore next */,
  arc: () => undefined /* v8 ignore next */ /* v8 ignore next */,
  arcTo: () => undefined /* v8 ignore next */ /* v8 ignore next */,
  moveTo: () => undefined /* v8 ignore next */ /* v8 ignore next */,
  lineTo: () => undefined /* v8 ignore next */ /* v8 ignore next */,
  quadraticCurveTo: () => undefined /* v8 ignore next */ /* v8 ignore next */,
  bezierCurveTo: () => undefined /* v8 ignore next */ /* v8 ignore next */,
  rect: () => undefined /* v8 ignore next */ /* v8 ignore next */,
  roundRect: () => undefined /* v8 ignore next */ /* v8 ignore next */,
  fill: () => undefined /* v8 ignore next */ /* v8 ignore next */,
  stroke: () => undefined /* v8 ignore next */ /* v8 ignore next */,
  clip: () => undefined /* v8 ignore next */ /* v8 ignore next */,
  fillText: () => undefined /* v8 ignore next */ /* v8 ignore next */,
  strokeText: () => undefined /* v8 ignore next */ /* v8 ignore next */,
  measureText: () => ({ width: 0 }) /* v8 ignore next */ /* v8 ignore next */,
  isPointInPath: () => false /* v8 ignore next */ /* v8 ignore next */,
  isPointInStroke: () => false /* v8 ignore next */ /* v8 ignore next */,
  createLinearGradient: () => ({
    addColorStop: () => undefined,
  }) /* v8 ignore next */ /* v8 ignore next */,
  createRadialGradient: () => ({
    addColorStop: () => undefined,
  }) /* v8 ignore next */ /* v8 ignore next */,
  createPattern: () => ({}) /* v8 ignore next */ /* v8 ignore next */,
}; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
if (typeof HTMLCanvasElement !== 'undefined') {
  /* v8 ignore next */ /* v8 ignore next */
  (HTMLCanvasElement as Object).prototype.getContext = () =>
    mockContext; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
(globalThis as Object).workerInstances = []; /* v8 ignore next */ /* v8 ignore next */
class MockWorker {
  /* v8 ignore next */ /* v8 ignore next */
  constructor() {
    /* v8 ignore next */ /* v8 ignore next */
    (globalThis as Object).workerInstances.push(this); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  onmessage: Object; /* v8 ignore next */ /* v8 ignore next */
  onerror: Object; /* v8 ignore next */ /* v8 ignore next */
  postMessage(data: Object) {
    /* v8 ignore next */ /* v8 ignore next */
    if (this.onmessage) {
      /* v8 ignore next */ /* v8 ignore next */
      this.onmessage({
        /* v8 ignore next */ /* v8 ignore next */
        data: {
          /* v8 ignore next */ /* v8 ignore next */
          type: 'PARSE_SUCCESS' /* v8 ignore next */ /* v8 ignore next */,
          layout: {
            /* v8 ignore next */ /* v8 ignore next */
            nodes: [
              /* v8 ignore next */ /* v8 ignore next */
              {
                id: '1',
                name: 'AddNode',
                opType: 'Add',
                type: 'node',
              } /* v8 ignore next */ /* v8 ignore next */,
              { id: 'input_X', name: 'X', type: 'input' } /* v8 ignore next */ /* v8 ignore next */,
              {
                id: 'output_Y',
                name: 'Y',
                type: 'output',
              } /* v8 ignore next */ /* v8 ignore next */,
              {
                id: 'constant_W',
                name: 'W',
                type: 'constant',
              } /* v8 ignore next */ /* v8 ignore next */,
            ] /* v8 ignore next */ /* v8 ignore next */,
            edges: [] /* v8 ignore next */ /* v8 ignore next */,
          } /* v8 ignore next */ /* v8 ignore next */,
          graph: {
            /* v8 ignore next */ /* v8 ignore next */
            nodes: [
              /* v8 ignore next */ /* v8 ignore next */
              {
                /* v8 ignore next */ /* v8 ignore next */
                name: 'AddNode' /* v8 ignore next */ /* v8 ignore next */,
                opType: 'Add' /* v8 ignore next */ /* v8 ignore next */,
                inputs: ['X', 'W'] /* v8 ignore next */ /* v8 ignore next */,
                outputs: ['Y'] /* v8 ignore next */ /* v8 ignore next */,
                attributes: {
                  attr1: { type: 'FLOAT', value: 1.0 },
                } /* v8 ignore next */ /* v8 ignore next */,
                domain: '' /* v8 ignore next */ /* v8 ignore next */,
              } /* v8 ignore next */ /* v8 ignore next */,
            ] /* v8 ignore next */ /* v8 ignore next */,
            tensors: {
              W: { name: 'W', dtype: 'float32', shape: [1], size: 1 },
            } /* v8 ignore next */ /* v8 ignore next */,
            inputs: [
              { name: 'X', dtype: 'float32', shape: [1] },
            ] /* v8 ignore next */ /* v8 ignore next */,
            outputs: [
              { name: 'Y', dtype: 'float32', shape: [1] },
            ] /* v8 ignore next */ /* v8 ignore next */,
            initializers: ['W'] /* v8 ignore next */ /* v8 ignore next */,
          } /* v8 ignore next */ /* v8 ignore next */,
        } /* v8 ignore next */ /* v8 ignore next */,
      }); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  terminate() {} /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
(globalThis as Object).Worker = MockWorker; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
if (typeof Blob === 'undefined') {
  /* v8 ignore next */ /* v8 ignore next */
  (globalThis as Object).Blob = class Blob {
    /* v8 ignore next */ /* v8 ignore next */
    constructor(public parts: Object[]) {} /* v8 ignore next */ /* v8 ignore next */
  }; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
if (typeof File === 'undefined') {
  /* v8 ignore next */ /* v8 ignore next */
  (globalThis as Object).File = class File extends Blob {
    /* v8 ignore next */ /* v8 ignore next */
    constructor(
      /* v8 ignore next */ /* v8 ignore next */
      parts: Object[] /* v8 ignore next */ /* v8 ignore next */,
      public name: string /* v8 ignore next */ /* v8 ignore next */,
    ) {
      /* v8 ignore next */ /* v8 ignore next */
      super(parts); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  }; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// @ts-ignore /* v8 ignore next */ /* v8 ignore next */
global.Path2D = class Path2D {
  /* v8 ignore next */ /* v8 ignore next */
  moveTo() {} /* v8 ignore next */ /* v8 ignore next */
  lineTo() {} /* v8 ignore next */ /* v8 ignore next */
  arc() {} /* v8 ignore next */ /* v8 ignore next */
  closePath() {} /* v8 ignore next */ /* v8 ignore next */
  bezierCurveTo() {} /* v8 ignore next */ /* v8 ignore next */
  rect() {} /* v8 ignore next */ /* v8 ignore next */
};
