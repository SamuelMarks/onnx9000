/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import tf from '../src/index.js';
import {
  Tensor,
  diag,
  cumsum,
  cumprod,
  eye,
  customGrad,
  grads,
  valueAndGrad,
  grad,
  model,
} from '../src/index.js';
import '../src/ui.js';

describe('TFJS Shim Consolidated Tests', () => {
  const a = tf.tensor([1, 2, 3, 4], [2, 2]);
  const b = tf.tensor([5, 6, 7, 8], [2, 2]);

  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ json: async () => ({}) }));
    document.body.innerHTML = '';
  });

  it('should cover numeric padding in all ops', () => {
    const x4d = tf.tensor(new Float32Array(16), [1, 4, 4, 1]);
    const w4d = tf.tensor(new Float32Array(4), [2, 2, 1, 1]);
    tf.conv2dTranspose(x4d, w4d, [1, 4, 4, 1], 1, 1); // numeric pad

    const x5d = tf.tensor(new Float32Array(64), [1, 4, 4, 4, 1]);
    tf.maxPool3d(x5d, 2, 1, 1); // numeric pad
    tf.avgPool3d(x5d, 2, 1, 1); // numeric pad

    (tf as any).pool(x4d, 2, 'max', 1, 1, 1); // numeric pad

    tf.conv2d(x4d, w4d, 1, 1); // numeric pad
    tf.depthwiseConv2d(x4d, w4d, 1, 1); // numeric pad
  });

  it('should cover max and mean with axis', () => {
    tf.max(a, 0);
    tf.mean(a, 1);
  });

  it('should cover tensor creation from TypedArray', () => {
    const t = tf.tensor(new Float32Array([1, 2, 3, 4]), [2, 2]);
    expect(t.size).toBe(4);
  });

  it('should cover image processing extras', () => {
    const img = tf.tensor(new Float32Array(1 * 4 * 4 * 3), [1, 4, 4, 3]);
    const boxes = tf.tensor2d([0, 0, 1, 1], [1, 4]);
    const boxInd = tf.tensor1d([0], 'int32');
    tf.image.cropAndResize(img, boxes, boxInd, [2, 2], 'bilinear');
    tf.image.cropAndResize(img, boxes, boxInd, [2, 2], 'nearest');

    tf.browser.fromPixels({ width: 2, height: 2 } as any);
  });

  it('should cover LayersModel and Layer', () => {
    const l = new (tf as any).layers.Layer({ name: 'test' });
    expect(l.getWeights()).toEqual([]);
    l.setWeights([]);

    const m = tf.sequential();
    m.add(tf.layers.dense({ units: 10, inputShape: [5] }));
    m.compile({});
    m.predict(a);
    m.evaluate(a, a);
    m.layers[0].getWeights();
  });

  it('should cover model and grad', () => {
    const m = model({ inputs: [], outputs: [] } as any);
    expect(m).toBeDefined();

    const f = (x: Tensor) => tf.sum(tf.mul(x, x));
    const gradFn = grad(f as any);
    const g = gradFn(tf.tensor1d([1, 2]), tf.tensor1d([1]));
    expect(g.size).toBe(2);
  });

  it('should cover grads and valueAndGrad', () => {
    const f = (x: Tensor) => tf.sum(tf.mul(x, x));
    const gradFn = grads(f as any);
    const g = gradFn(tf.tensor1d([1, 2]));
    expect(g[0].size).toBe(2);

    const vgFn = valueAndGrad(f as any);
    const { value, grads: gs } = vgFn(tf.tensor1d([1, 2]));
    expect(value).toBeDefined();
    expect(gs.length).toBe(1);
  });

  it('should cover customGrad', () => {
    const f = (x: Tensor) => ({ value: x, gradFunc: (dy: Tensor) => dy });
    const g = customGrad(f as any);
    g(a);
  });

  it('should cover diag and eye', () => {
    expect(diag).toBeDefined();
    diag(tf.tensor1d([1, 2]));
    eye(3, 4);
  });

  it('should cover cumulative ops surgically', () => {
    const x = tf.tensor1d([1, 2, 3]);
    cumsum(x, 0, true, true);
    cumsum(x, 0, false, false); // inclusive
    cumprod(x, 0, true, true);
    cumprod(x, 0, false, false); // inclusive
  });

  it('should cover math, unary and reductions', () => {
    tf.add(a, b);
    tf.sub(a, b);
    tf.mul(a, b);
    tf.div(a, b);
    tf.divNoNan(a, b);
    tf.floorDiv(a, b);
    tf.maximum(a, b);
    tf.minimum(a, b);
    tf.mod(a, b);
    tf.pow(a, b);
    tf.squaredDifference(a, b);
    tf.abs(a);
    tf.acos(a);
    tf.asin(a);
    tf.atan(a);
    tf.ceil(a);
    tf.cos(a);
    tf.cosh(a);
    tf.erf(a);
    tf.exp(a);
    tf.floor(a);
    tf.log(a);
    tf.neg(a);
    tf.round(a);
    tf.sign(a);
    tf.sin(a);
    tf.sinh(a);
    tf.sqrt(a);
    tf.square(a);
    tf.tan(a);

    tf.matMul(a, b);
    tf.dot(a, b);
    tf.outerProduct(tf.tensor1d([1, 2]), tf.tensor1d([3, 4]));
    tf.norm(a);

    tf.argMax(a, 0);
    tf.argMin(a, 1);
    tf.max(a, 0);
    tf.mean(a, 1);
    tf.min(a);
    tf.prod(a);
    tf.sum(a);
    tf.all(a);
    tf.any(a);
    tf.logSumExp(a);
  });

  it('should cover transformations and slice/gather', () => {
    tf.reshape(a, [4]);
    tf.reshape(a, [-1, 1]); // inference
    tf.expandDims(a, 0);
    tf.squeeze(tf.tensor([1], [1, 1]));
    tf.transpose(a);
    tf.concat([a, b], 0);
    tf.split(a, 2, 0);
    tf.split(a.flatten(), [2, 2]); // array splits
    tf.stack([a, b]);
    tf.unstack(a);
    tf.tile(a, [2, 2]);
    tf.reverse(a);

    tf.pad(a, [
      [1, 1],
      [1, 1],
    ]);
    tf.pad(tf.tensor1d([1]), [[1, 1]], 0); // 1D pad
    tf.slice(a, [0, 0], [1, 1]);
    tf.stridedSlice(a, [0, 0], [2, 2], [1, 1]);

    const idx = tf.tensor([0, 1], [2], 'int32');
    tf.gather(a, idx);
    tf.gatherND(a, idx);
    tf.scatterND(idx, a, [4, 4]);
    tf.tensorScatterUpdate(a, idx, a);

    tf.batchToSpaceND(
      a,
      [1, 1],
      [
        [0, 0],
        [0, 0],
      ],
    );
    tf.spaceToBatchND(
      a,
      [1, 1],
      [
        [0, 0],
        [0, 0],
      ],
    );
    tf.depthToSpace(a, 2);
    tf.spaceToDepth(a, 2);
  });

  it('should cover activations and complex conv/pool', () => {
    tf.relu(a);
    tf.relu6(a);
    tf.leakyRelu(a);
    tf.elu(a);
    tf.selu(a);
    tf.sigmoid(a);
    tf.softmax(a);
    tf.logSoftmax(a);
    tf.softplus(a);
    tf.localResponseNormalization(a);

    const x4d = tf.tensor(new Float32Array(16), [1, 4, 4, 1]);
    const w4d = tf.tensor(new Float32Array(4), [2, 2, 1, 1]);
    tf.conv2d(x4d, w4d, 1, 'same');
    tf.depthwiseConv2d(x4d, w4d, 1, 'valid');
    tf.separableConv2d(x4d, w4d, w4d, 1, 'same');
    tf.conv2dTranspose(x4d, w4d, [1, 4, 4, 1], 1, 'same');

    tf.maxPool(x4d, 2, 2, 'valid');
    tf.avgPool(x4d, 2, 2, 'valid');

    const x3d = tf.tensor(new Float32Array(8), [2, 2, 2]);
    (tf as any).pool(x3d, 2, 'max', 'same'); // 3D input pool

    const x5d = tf.tensor(new Float32Array(64), [1, 4, 4, 4, 1]);
    const w5d = tf.tensor(new Float32Array(8), [2, 2, 2, 1, 1]);
    tf.maxPool3d(x5d, 2, 2, 'same');
    tf.avgPool3d(x5d, 2, 2, 'same');
    tf.conv3dTranspose(x5d, w5d, [1, 4, 4, 4, 1], 2, 'same');
  });

  it('should cover logical ops and condition', () => {
    const boolA = tf.tensor([true, false, true, false], [2, 2], 'bool');
    tf.equal(a, b);
    tf.notEqual(a, b);
    tf.less(a, b);
    tf.greater(a, b);
    tf.lessEqual(a, a);
    tf.greaterEqual(a, a);

    tf.logicalAnd(boolA, boolA);
    tf.logicalOr(boolA, boolA);
    tf.logicalNot(boolA);
    tf.logicalXor(boolA, boolA);
    tf.where(boolA, a, b);

    tf.booleanMaskAsync(a, boolA);
    tf.whereAsync(boolA);
  });

  it('should cover image processing branches', async () => {
    const img = tf.tensor(new Float32Array(16), [1, 2, 2, 4]);
    tf.image.resizeBilinear(img, [4, 4], true, true);
    tf.image.resizeNearestNeighbor(img, [4, 4], true, true);

    const boxes = tf.tensor2d([0, 0, 1, 1, 0.1, 0.1, 1.1, 1.1], [2, 4]);
    const scores = tf.tensor1d([0.9, 0.8]);
    tf.image.nonMaxSuppression(boxes, scores, 2, 0.5);
    await tf.image.nonMaxSuppressionAsync(boxes, scores, 2, 0.5);
    tf.image.nonMaxSuppressionWithScore(boxes, scores, 2, 0.5);
    tf.image.flipLeftRight(img);

    const pixels = { data: new Uint8Array(16), width: 2, height: 2 };
    tf.browser.fromPixels(pixels as any);
    await tf.browser.toPixels(tf.tensor1d([0, 0.5, 1]));
  });

  it('should cover initializers and random', () => {
    tf.ones([2, 2]);
    tf.zeros([2, 2]);
    tf.fill([2, 2], 5);
    tf.eye(2);
    tf.linspace(0, 1, 5);
    tf.range(0, 5);
    tf.onesLike(a);
    tf.zerosLike(a);

    tf.randomUniform([2, 2], 0, 1, 'float32', 42);
    tf.randomNormal([2, 2], 0, 1, 'float32', 42);
    tf.truncatedNormal([2, 2]);
    tf.randomGamma([2, 2], 1);
    tf.multinomial(tf.tensor2d([0.5, 0.5], [1, 2]), 1, 42);
  });

  it('should cover models and training', async () => {
    const v = tf.variable(a);
    v.assign(b);

    const ml = new tf.GraphModel('mock-url');
    ml.predict(a);
    ml.predict([a, a]);
    ml.predict({ in: a });
    ml.execute(a);
    await ml.executeAsync(a);
    ml.dispose();

    await tf.loadGraphModel('url');
    await tf.loadLayersModel('url');

    const sequential = tf.sequential();
    sequential.add(tf.layers.dense({ units: 10, inputShape: [5] }));
    sequential.compile({});
    sequential.predict(a);
    sequential.evaluate(a, a);

    const adam = tf.train.adam(0.1);
    adam.applyGradients([a]);
    adam.applyGradients({ a: a });

    const sgd = tf.train.sgd(0.1);
    sgd.applyGradients([a]);
    sgd.applyGradients({ a: a });
  });

  it('should cover losses, metrics and special ops', () => {
    const yTrue = tf.tensor1d([1, 0]);
    const yPred = tf.tensor1d([0.8, 0.2]);
    const w = tf.tensor1d([1, 1]);
    tf.losses.meanSquaredError(yTrue, yPred, w);
    tf.losses.sigmoidCrossEntropy(yTrue, yPred, w, 0.1);

    tf.metrics.binaryAccuracy(yTrue, yPred);
    tf.metrics.categoricalAccuracy(yTrue.reshape([1, 2]), yPred.reshape([1, 2]));
    tf.metrics.categoricalAccuracy(
      tf.tensor2d([1, 0, 0, 1], [2, 2]),
      tf.tensor2d([0.1, 0.9, 0.8, 0.2], [2, 2]),
    );

    tf.clipByValue(a, 0, 1);
    tf.einsum('ij,jk->ik', a, b);
    tf.einsum('i->i', a); // single tensor

    tf.string.decodeString(new Uint8Array([72, 73]));
    tf.string.encodeString('HI');
    tf.string.stringSplit(tf.tensor1d(['a,b']), ',');
    tf.string.stringToHashBucketFast(tf.tensor1d(['hello']), 10);

    const complex = tf.complex(tf.tensor1d([1, 2]), tf.tensor1d([3, 4]));
    tf.real(complex);
    tf.imag(complex);

    tf.spectral.rfft(tf.tensor1d([1, 2, 3, 4]));
    tf.spectral.rfft(tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]));

    tf.signal.stft(tf.tensor1d(new Float32Array(20).fill(1)), 8, 4, 8, (n) => Math.sin(n));
  });

  it('should cover utilities and lifecycle', async () => {
    await tf.ready();
    tf.enableProdMode();
    tf.enableDebugMode();
    tf.env().set('DEBUG', true);
    tf.memory();
    await tf.profile(() => tf.add(a, b));
    await tf.time(() => tf.add(a, b));
    tf.disposeVariables();
    tf.print(a);
    tf.util.createShuffledIndices(5);
    await tf.util.fetch('http://localhost');
    await tf.setDevice('cpu');
    await tf.nextFrame();
  });

  it('should cover UI demo element', () => {
    const el = document.createElement('tfjs-shim-demo') as any;
    document.body.appendChild(el);
    expect(el.shadowRoot.innerHTML).toContain('TF.js vs onnx9000 Shim');

    const runBtn = el.shadowRoot.querySelector('#run-btn');
    runBtn.click();
    expect(el.shadowRoot.querySelector('#results').textContent).toContain('Results match!');
  });

  it('should cover branch coverage missing parts', () => {
    // tf.metrics.categoricalAccuracy without batch dimension
    const yTrue = tf.tensor1d([0, 1, 0]);
    const yPred = tf.tensor1d([0.1, 0.9, 0.0]);
    const acc = tf.metrics.categoricalAccuracy(yTrue, yPred);
    expect(acc.dataSync()[0]).toBe(1);

    // tf.signal.stft with fftLength
    const signal = tf.tensor1d([1, 2, 3, 4]);
    const stftRes = tf.signal.stft(signal, 2, 2, 4);
    expect(stftRes.shape).toEqual([2, 4]);

    // tf.spectral.rfft with fftLength and 2D shape
    const rfftInput = tf.tensor2d(
      [
        [1, 2],
        [3, 4],
      ],
      [2, 2],
    );
    const rfftRes = tf.spectral.rfft(rfftInput, 4);
    expect(rfftRes.shape).toEqual([2, 3]);

    // tf.fill string fallback
    const fillRes = tf.fill([2], 'test');
    expect(fillRes.dtype).toBe('string');
  });

  it('should cover tf.io mock loaders', async () => {
    // Mock fetch for browserHTTPRequest
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ mock: 'json' }),
    } as any);

    const requestLoader = tf.io.browserHTTPRequest('http://localhost');
    const reqResult = await requestLoader.load();
    expect(reqResult).toEqual({ mock: 'json' });

    // Test failing fetch
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
    } as any);
    await expect(tf.io.browserHTTPRequest('http://localhost').load()).rejects.toThrow();

    // Test browserFiles
    const fileLoader = tf.io.browserFiles([new File([''], 'model.json')]);
    const fileResult = await fileLoader.load();
    expect(fileResult.modelTopology).toBeDefined();

    // Test empty browserFiles
    await expect(tf.io.browserFiles([]).load()).rejects.toThrow();
  });
});
