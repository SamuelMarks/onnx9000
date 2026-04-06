import { describe, it, expect, vi, beforeEach } from 'vitest';
import { PolyfillMLGraphBuilder } from '../src/builder';
import { PolyfillMLContext } from '../src/context';
import { PolyfillMLOperand } from '../src/operand';
import { PolyfillMLTensor } from '../src/tensor';
import { PolyfillMLGraph } from '../src/graph';
import { PolyfillML } from '../src/index';

// Mock the backend
vi.mock('@onnx9000/backend-web', () => {
  return {
    InferenceSession: vi.fn().mockImplementation(() => ({
      run: vi.fn().mockResolvedValue({
        out: { data: new Float32Array([42]) },
      }),
    })),
    WebGPUProvider: vi.fn(),
    WasmProvider: vi.fn(),
  };
});

describe('WebNN Polyfill Exhaustive Coverage', () => {
  const context = new PolyfillMLContext({ deviceType: 'cpu' });
  const builder = new PolyfillMLGraphBuilder(context);

  const desc = { dataType: 'float32' as const, dimensions: [1] };
  const a = builder.input('a', desc);

  it('should cover all builder mapping methods and branches', async () => {
    // Math & Unary
    builder.add(a, a);
    builder.sub(a, a);
    builder.mul(a, a);
    builder.div(a, a);
    builder.max(a, a);
    builder.min(a, a);
    builder.pow(a, a);
    builder.abs(a);
    builder.ceil(a);
    builder.floor(a);
    builder.exp(a);
    builder.log(a);
    builder.cos(a);
    builder.sin(a);
    builder.tan(a);
    builder.acos(a);
    builder.asin(a);
    builder.atan(a);
    builder.sqrt(a);
    builder.erf(a);
    builder.sign(a);
    builder.neg(a);

    // Activations
    builder.relu(a);
    builder.sigmoid(a);
    builder.tanh(a);
    builder.softmax(a);
    builder.softmax(a, 0); // With axis
    builder.softplus(a);
    builder.softsign(a);
    builder.elu(a, { alpha: 0.1 });
    builder.leakyRelu(a, { alpha: 0.1 });
    builder.prelu(a, a);
    builder.gelu(a);
    builder.hardSigmoid(a, { alpha: 0.1, beta: 0.1 });
    builder.hardSwish(a);
    builder.clamp(a, { minValue: a, maxValue: a });
    builder.clamp(a, { maxValue: a });
    builder.linear(a, { alpha: 2.0, beta: 1.0 });
    builder.linear(a); // No options

    // Linear
    builder.gemm(a, a, { c: a, aTranspose: true, bTranspose: true, alpha: 0.5, beta: 0.5 });
    builder.matmul(a, a);

    // Tensor ops
    builder.transpose(a, { permutation: [0] });
    builder.transpose(a);
    builder.reshape(a, [1]);
    builder.slice(a, [0], [1], { axes: [0], strides: [1] });
    builder.slice(a, [0], [1], { strides: [1] });
    builder.split(a, 1, { axis: 0 });
    builder.split(a, [1]); // Array splits
    builder.split(a, 1);
    builder.concat([a, a], 0);
    builder.pad(a, [0, 0], { mode: 'edge', value: 1.0 });
    builder.gather(
      a,
      builder.constant({ dataType: 'int32', dimensions: [1] }, new Int32Array([0])),
      { axis: 0 },
    );
    builder.gatherNd(a, a);
    builder.scatterNd(a, a, { shape: [1] });
    builder.gatherElements(a, a, { axis: 0 });
    builder.expand(a, [1]);
    builder.cast(a, 'int32');
    builder.triangular(a, { diagonal: 0, upper: true });

    // Pool/Conv
    const x4d = builder.input('x4d', { dataType: 'float32', dimensions: [1, 1, 4, 4] });
    const w4d = builder.constant(
      { dataType: 'float32', dimensions: [1, 1, 3, 3] },
      new Float32Array(9),
    );
    builder.conv2d(x4d, w4d, { padding: [1, 1, 1, 1], autoPad: 'same-upper', bias: a });
    builder.convTranspose2d(x4d, w4d, {
      dilations: [1, 1],
      groups: 1,
      bias: a,
      outputPadding: [0, 0],
    });
    builder.maxPool2d(x4d, {
      windowDimensions: [2, 2],
      strides: [1, 1],
      padding: [0, 0, 0, 0],
      roundingType: 'ceil',
    });
    builder.averagePool2d(x4d, {
      windowDimensions: [2, 2],
      strides: [1, 1],
      padding: [0, 0, 0, 0],
      roundingType: 'ceil',
    });
    builder.l2Pool2d(x4d, { windowDimensions: [2, 2], strides: [1, 1], padding: [0, 0, 0, 0] });

    // Reduction
    builder.reduceMean(a, { keepDimensions: false });
    builder.reduceSum(a, { axes: [0] });
    builder.reduceMax(a);
    builder.reduceMin(a);
    builder.reduceProduct(a);
    builder.reduceL1(a);
    builder.reduceL2(a);
    builder.reduceLogSumExp(a);
    builder.argMax(a, { axes: [0] });
    builder.argMin(a);

    // Normalization
    builder.batchNormalization(a, a, a, { epsilon: 1e-4 });
    builder.instanceNormalization(a);
    builder.layerNormalization(a);
    builder.l2Normalization(a);

    // Logical
    builder.equal(a, a);
    builder.greater(a, a);
    builder.greaterOrEqual(a, a);
    builder.lesser(a, a);
    builder.lesserOrEqual(a, a);
    builder.logicalNot(a);
    builder.logicalAnd(a, a);
    builder.logicalOr(a, a);
    builder.logicalXor(a, a);
    const cond = builder.input('cond', { dataType: 'uint8', dimensions: [1] });
    builder.where(cond, a, a);

    // Cell ops
    builder.lstmCell(a, a, a, a, a, 1);
    builder.gruCell(a, a, a, a, 1);

    // Transformers/Quant
    builder.scaledDotProductAttention(a, a, a, {});
    builder.quantizeLinear(a, a, a);
    builder.dequantizeLinear(a, a, a);
    builder.bitwiseAnd(a, a);
    builder.shiftRightLogical(a, a);

    // build with naming mismatch to trigger Identity
    await builder.build({ result: a });
  });

  it('should cover context and graph edge cases', async () => {
    const gpuCtx = new PolyfillMLContext({ deviceType: 'gpu' });
    const unknownCtx = new PolyfillMLContext({ deviceType: 'npu' as 'cpu' });

    context.opSupportLimits();

    const mockOnnxGraph = {
      outputs: [{ name: 'out' }],
      inputs: [{ name: 'a', shape: [1], dtype: 'float32' }],
      addNode: vi.fn(),
      addTensor: vi.fn(),
    };
    type GraphType = ConstructorParameters<typeof PolyfillMLGraph>[0];
    const mockGraph = new PolyfillMLGraph(mockOnnxGraph as object as GraphType);

    const tIn = await context.createTensor({ dataType: 'float32', dimensions: [1] });
    const tOut = await context.createTensor({ dataType: 'float32', dimensions: [1], usage: 1 });

    // dispatch/compute coverage with mocked session
    await context.compute(mockGraph, { a: new Float32Array([1]) }, { out: new Float32Array([0]) });
    await context.dispatch(mockGraph, { a: tIn }, { out: tOut });

    // Invalid inputs
    await expect(
      context.compute(mockGraph, { invalid: new Float32Array(1) }, {}),
    ).rejects.toThrow();
    await expect(context.dispatch(mockGraph, { invalid: tIn }, {})).rejects.toThrow();

    // graph lifecycle
    mockGraph.destroy();
  });

  it('should cover input validation gaps', () => {
    expect(() => builder.input('fail', { dataType: 'float32', dimensions: [-1] })).toThrow();
    expect(() =>
      builder.input('fail', { dataType: 'invalid' as 'float32', dimensions: [1] }),
    ).toThrow();
  });

  it('should cover MLTensor gaps', () => {
    const tensor = new PolyfillMLTensor({ dataType: 'float32', dimensions: [1] });
    expect(tensor.dataType).toBe('float32');
    expect(tensor.dimensions).toEqual([1]);
    tensor.destroy();

    const mockDevice = {
      createBuffer: vi.fn().mockReturnValue({ destroy: vi.fn() }),
    };
    const gpuTensor = new PolyfillMLTensor({ dataType: 'int32', dimensions: [4] }, mockDevice);
    gpuTensor.destroy();
  });
});
