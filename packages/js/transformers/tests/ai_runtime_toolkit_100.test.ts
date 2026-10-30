import { describe, expect, it, vi } from 'vitest';
import { BaseTensor } from '../../array/src/index.js';
import {
  DDIMScheduler,
  DDPMScheduler,
  EulerDiscreteScheduler,
  LCMScheduler,
} from '../../diffusers/src/schedulers.js';
import { DynamicBatcher } from '../../serve/src/batcher.js';
import { ModelEnsemble } from '../../serve/src/ensemble.js';
import { createServer } from '../../serve/src/index.js';
import { createLambdaHandler } from '../../serve/src/lambda.js';
import { Logger, LogLevel } from '../../serve/src/logger.js';
import { serveNode } from '../../serve/src/node.js';
import { Router } from '../../serve/src/router.js';
import {
  add,
  all,
  any,
  avgPool3d,
  browser,
  conv2d,
  conv2dTranspose,
  conv3d,
  conv3dTranspose,
  depthwiseConv2d,
  divNoNan,
  elu,
  env,
  equal,
  expandDims,
  grad,
  greater,
  greaterEqual,
  image,
  layers,
  leakyRelu,
  less,
  lessEqual,
  logicalXor,
  logSumExp,
  matMul,
  max,
  maxPool,
  maxPool3d,
  mean,
  metrics,
  min,
  norm,
  notEqual,
  pool,
  prod,
  selu,
  signal,
  slice,
  spectral,
  squeeze,
  sum,
  Tensor as TfTensor,
  tensor1d,
  unstack,
  util,
} from '../../tfjs-shim/src/index.js';
import { ArrayAPI } from '../src/array/index.js';
import {
  AutomaticSpeechRecognitionPipeline,
  Callable,
  FeatureExtractionPipeline,
  ImageClassificationPipeline,
  ImageSegmentationPipeline,
  ObjectDetectionPipeline,
  pipeline,
  QuestionAnsweringPipeline,
  TextClassificationPipeline,
  TextGenerationPipeline,
  TokenClassificationPipeline,
  ZeroShotClassificationPipeline,
} from '../src/pipelines/index.js';
import {
  AutoProcessor,
  BaseImageProcessor,
  ONNX9000Audio,
  ONNX9000Image,
  SequenceFeatureExtractor,
} from '../src/processors/index.js';

describe('Phase 6.3.6: Exhaustive 100% coverage for TS AI/Runtime/Toolkit packages', () => {
  it('covers @onnx9000/transformers array, pipelines, and processors', async () => {
    // ArrayAPI
    expect(ArrayAPI.add([1, 2], [3])).toEqual([4, 2]);
    expect(ArrayAPI.log_softmax([1, 2])).toHaveLength(2);
    expect(ArrayAPI.get_top_k([1, 5, 3], 2)).toEqual({ values: [5, 3], indices: [1, 2] });
    expect(ArrayAPI.dot_product([1, 2], [3])).toBe(3);
    expect(ArrayAPI._maybeDispatchWasm(new Array(10001).fill(0))).toBe(true);
    expect(ArrayAPI._maybeDispatchWasm([1, 2])).toBe(false);
    const tObj = { dims: [2, 2], transposed: false, axes: undefined as number[] | undefined };
    expect(ArrayAPI.view(tObj, [4]).dims).toEqual([4]);
    expect(ArrayAPI.reshape(tObj, [1, 4]).dims).toEqual([1, 4]);
    expect(ArrayAPI.transpose(tObj, [1, 0]).transposed).toBe(true);
    const f32 = ArrayAPI.toFloat32Array([1, 2]);
    expect(ArrayAPI.fromFloat32Array(f32)).toEqual([1, 2]);
    expect(ArrayAPI.toJSON([1, 2])).toEqual([1, 2]);
    expect(ArrayAPI.toJSON('not_array')).toEqual([]);
    expect(ArrayAPI.fromJSON([1, 2])).toEqual([1, 2]);
    expect(ArrayAPI.slice([1, 2, 3, 4], 1, 3)).toEqual([2, 3]);
    expect(ArrayAPI.getStrided([1, 2, 3, 4], 2)).toEqual([1, 3]);
    expect(ArrayAPI.getStrided(123, 2)).toBe(123);
    expect(ArrayAPI.erf(1.0)).toBeGreaterThan(0);
    expect(ArrayAPI.erf(-1.0)).toBeLessThan(0);

    // Callable & Pipelines
    const baseCall = new Callable();
    expect(baseCall._call()).toBeUndefined();
    expect(null instanceof Callable).toBe(false);
    expect(123 instanceof Callable).toBe(false);
    expect({} instanceof Callable).toBe(false);
    expect((() => {}) instanceof Callable).toBe(false);
    expect(baseCall instanceof Callable).toBe(true);

    const fePipe = new FeatureExtractionPipeline(null, null, null, {});
    expect(await fePipe(['a', 'b'])).toEqual(['a', 'b']);

    const tcPipe = new TextClassificationPipeline(null, null, null, {});
    expect(await tcPipe.postprocess({ a: 1 }, { return_tensors: true })).toEqual({ a: 1 });
    expect(await tcPipe.postprocess({ a: 1 }, { top_k: 2 })).toEqual([
      { label: 'positive', score: 0.99, top_k: 2 },
    ]);

    expect(
      await new TokenClassificationPipeline(null, null, null, {}).postprocess({}),
    ).toHaveLength(1);
    expect((await new QuestionAnsweringPipeline(null, null, null, {}).postprocess({})).answer).toBe(
      'hello',
    );
    expect(
      (await new ZeroShotClassificationPipeline(null, null, null, {}).postprocess({})).sequence,
    ).toBe('text');

    const tgPipe = new TextGenerationPipeline(null, null, null, {});
    const streamOut = await tgPipe('hello', { stream: true });
    const chunks: string[] = [];
    for await (const ch of streamOut) chunks.push(ch);
    expect(chunks).toEqual(['hello [GENERATED]']);
    expect(await tgPipe('hello')).toEqual([{ generated_text: 'hello [GENERATED]' }]);

    expect(
      await new ImageClassificationPipeline(null, null, null, {}).postprocess({}),
    ).toHaveLength(1);
    const odPipe = new ObjectDetectionPipeline(null, null, null, {});
    expect((await odPipe.postprocess({}, { threshold: 0.7 })).threshold).toBe(0.7);
    expect((await odPipe.postprocess({})).threshold).toBe(0.5);
    const boxes = [[0.1, 0.2, 0.3, 0.4]];
    odPipe.denormalizeBbox(boxes);
    expect(boxes[0]?.[0]).toBeCloseTo(25.5);

    // BaseTensor.numpy() and DDIM/DDPM/Euler/LCM schedulers with both number[] and Float32Array sample
    const bt = new BaseTensor('bt', [1], 'float32');
    expect(bt.numpy()).toBeNull();
    expect(new DDIMScheduler().step([0.1, 0.2], 10, [1.0, 2.0])).toHaveLength(2);
    expect(
      new DDIMScheduler().step(new Float32Array([0.1, 0.2]), 10, new Float32Array([1.0, 2.0])),
    ).toHaveLength(2);
    expect(new DDPMScheduler().step([0.1, 0.2], 10, [1.0, 2.0])).toHaveLength(2);
    expect(new EulerDiscreteScheduler().step([0.1, 0.2], 10, [1.0, 2.0])).toHaveLength(2);
    expect(
      new EulerDiscreteScheduler().step(
        new Float32Array([0.1, 0.2]),
        10,
        new Float32Array([1.0, 2.0]),
      ),
    ).toHaveLength(2);
    expect(new LCMScheduler().step([0.1, 0.2], 10, [1.0, 2.0])).toHaveLength(2);
    expect(await new ImageSegmentationPipeline(null, null, null, {}).postprocess({})).toHaveLength(
      1,
    );
    expect(
      (await new AutomaticSpeechRecognitionPipeline(null, null, null, {}).postprocess({})).text,
    ).toBe('speech text');

    const allTasks = [
      'feature-extraction',
      'text-classification',
      'token-classification',
      'question-answering',
      'zero-shot-classification',
      'translation',
      'summarization',
      'text-generation',
      'text2text-generation',
      'fill-mask',
      'image-classification',
      'object-detection',
      'zero-shot-image-classification',
      'image-segmentation',
      'depth-estimation',
      'image-to-image',
      'audio-classification',
      'automatic-speech-recognition',
      'text-to-speech',
      'document-question-answering',
      'visual-question-answering',
      'image-feature-extraction',
    ];
    for (const task of allTasks) {
      const p1 = await pipeline(task, 'm1');
      const p2 = await pipeline(task, 'm1');
      expect(p1).toBe(p2);
    }

    // Processors
    expect((await ONNX9000Image.fromURL('http://x')).data).toBe('http://x');
    expect(ONNX9000Image.fromBase64('abc').data).toBe('abc');
    expect((await ONNX9000Audio.fromURL('http://a')).data).toBe('http://a');
    expect(ONNX9000Audio.fromBlob(new Blob(['a'])).data).toBeInstanceOf(Blob);

    const imgProc = new BaseImageProcessor({
      do_resize: true,
      do_center_crop: true,
      do_random_crop: true,
      do_pad: true,
      do_rescale: true,
      do_normalize: true,
    });
    const rawImg = { data: new Uint8Array([1, 2, 3]) };
    await imgProc.process([rawImg]);
    expect((rawImg as { optimized?: boolean }).optimized).toBe(true);
    await imgProc.process('single_img', {
      image_mean: [0.1, 0.1, 0.1],
      image_std: [0.2, 0.2, 0.2],
    });
    expect(imgProc.wasmBilinearResize('x')).toBe('x');
    expect(imgProc.wasmBicubicResize('x')).toBe('x');
    expect(imgProc.wasmNearestNeighborResize('x')).toBe('x');
    expect(imgProc.webgpuNormalizeShader('x')).toBe('x');
    expect(imgProc.webgpuResizeShader('x')).toBe('x');

    const mockCtx = {
      strokeStyle: '',
      lineWidth: 0,
      strokeRect: vi.fn(),
      createImageData: () => ({ data: new Uint8Array(4) }),
      putImageData: vi.fn(),
    };
    const mockCanvas = {
      width: 2,
      height: 2,
      getContext: () => mockCtx,
    };
    BaseImageProcessor.drawBoundingBoxes(mockCanvas, [
      Object.assign([0, 0, 10, 10], { color: 'blue' }),
      [1, 1, 5, 5],
    ]);
    BaseImageProcessor.drawBoundingBoxes({ getContext: () => null }, []);
    BaseImageProcessor.drawBoundingBoxes({}, []);
    BaseImageProcessor.drawSegmentationMask(mockCanvas, [1, 2, 3, 4]);
    BaseImageProcessor.drawSegmentationMask({ getContext: () => null }, [1]);
    BaseImageProcessor.drawSegmentationMask({}, null);

    const seqExt = new SequenceFeatureExtractor({ do_pad: true, do_truncate: true });
    await seqExt.process(['aud1', 'aud2']);
    await seqExt.process('aud_single');
    expect(seqExt.wasmWindowing('aud', 'hann')).toBe('aud');
    expect(seqExt.chunkWaveform('aud')).toEqual(['aud']);
    expect(seqExt.applyVAD('aud')).toBe('aud');

    const autoProc = await AutoProcessor.fromPretrained('m');
    expect(await autoProc.process('image', { return_tensors: 'np' })).toHaveProperty(
      'pixel_values',
    );
  });

  it('covers @onnx9000/tfjs-shim layers, tensor, model, and ops', async () => {
    // layers.ts
    expect(layers.conv2d({ units: 4 })).toBeDefined();
    expect(layers.maxPooling2d({})).toBeDefined();
    expect(layers.flatten({})).toBeDefined();
    expect(layers.dropout({})).toBeDefined();
    expect(layers.batchNormalization({})).toBeDefined();
    expect(layers.reLU({})).toBeDefined();

    // tensor.ts
    env().set('k1', 'v1');
    expect(env().get('k1')).toBe('v1');
    expect(env().get('missing_key')).toBeNull();
    const emptyStrT = new TfTensor([1], 'string', ['']);
    expect(await emptyStrT.data()).toEqual(['']);
    expect(emptyStrT.dataSync()).toEqual(['']);
    const f32T = new TfTensor([1], 'float32', [1]);
    expect(await f32T.data()).toBeInstanceOf(Float32Array);
    const unkDtypeT = new TfTensor([1], 'custom' as never, [1]);
    expect(await unkDtypeT.data()).toBeInstanceOf(Float32Array);
    expect(unkDtypeT.dataSync()).toBeInstanceOf(Float32Array);
    expect(tensor1d([]).shape).toEqual([1]);
    expect(expandDims(new TfTensor([2, 2], 'float32', [1, 2, 3, 4]), -1).shape).toEqual([2, 2, 1]);
    expect(squeeze(new TfTensor([1, 2, 1], 'float32', [1, 2]), [0]).shape).toEqual([2, 1]);
    const enc = util.encodeString('hi');
    expect(util.decodeString(enc)).toBe('hi');

    // model.ts: fromPixels, resizeBilinear, resizeNearestNeighbor (both alignCorners true and false), cropAndResize, nonMaxSuppression, grad with and without dy, categoricalAccuracy with 0D
    expect(browser.fromPixels({ width: 0, height: 0 } as never).shape).toEqual([1, 1, 3]);
    const img4d = new TfTensor([1, 2, 2, 1], 'float32', [1, 2, 3, 4]);
    expect(image.resizeBilinear(img4d, [2, 2], true, true).shape).toEqual([1, 2, 2, 1]);
    expect(image.resizeBilinear(img4d, [1, 1], true, false).shape).toEqual([1, 1, 1, 1]);
    expect(image.resizeNearestNeighbor(img4d, [2, 2], true, true).shape).toEqual([1, 2, 2, 1]);
    expect(image.resizeNearestNeighbor(img4d, [1, 1], false, false).shape).toEqual([1, 1, 1, 1]);
    expect(
      image.cropAndResize(
        img4d,
        new TfTensor([2, 4], 'float32', [-1, -1, 2, 2, 0, -1, 1, 2]),
        new TfTensor([2], 'int32', [0, 0]),
        [1, 1],
        'nearest',
      ).shape,
    ).toEqual([2, 1, 1, 1]);
    expect(
      image.nonMaxSuppression(
        new TfTensor([3, 4], 'float32', [0, 0, 1, 1, 0, 0, 0, 0, 2, 2, 3, 3]),
        new TfTensor([3], 'float32', [0.9, 0.8, 0.7]),
        1,
      ).shape,
    ).toEqual([1]);
    expect(
      image.nonMaxSuppression(
        new TfTensor([2, 4], 'float32', [0, 0, 0, 0, 0, 0, 0, 0]),
        new TfTensor([2], 'float32', [0.9, 0.8]),
        2,
      ).shape,
    ).toEqual([2]);
    const gFn = grad((x) => x);
    expect(gFn(new TfTensor([2], 'float32', [1, 2])).shape).toEqual([2]);
    expect(
      gFn(new TfTensor([2], 'float32', [1, 2]), new TfTensor([2], 'float32', [2, 3])).shape,
    ).toEqual([2]);
    expect(
      metrics.categoricalAccuracy(
        new TfTensor([], 'float32', [1]),
        new TfTensor([], 'float32', [1]),
      ).nData,
    ).toEqual([1]);

    // ops/Matrices.ts: matMul transposeA/transposeB and mismatch, norm keepDims, stft with and without fftLength, rfft 0D and fftLength
    const m2x2 = new TfTensor([2, 2], 'float32', [1, 2, 3, 4]);
    expect(matMul(m2x2, m2x2, true, true).shape).toEqual([2, 2]);
    expect(() => matMul(new TfTensor([2, 3], 'float32', new Array(6).fill(1)), m2x2)).toThrow();
    expect(norm(m2x2, 'euclidean', undefined, true).shape).toEqual([1, 1]);
    expect(signal.stft(new TfTensor([4], 'float32', [1, 2, 3, 4]), 2, 1, 4).shape).toEqual([3, 4]);
    expect(signal.stft(new TfTensor([4], 'float32', [1, 2, 3, 4]), 2, 1).shape).toEqual([3, 2]);
    expect(spectral.rfft(new TfTensor([], 'float32', [1])).shape).toEqual([1]);
    expect(spectral.rfft(new TfTensor([2], 'float32', [1, 2]), 4).shape).toEqual([3]);

    // ops/math.ts: broadcast a larger than b and b larger than a, divNoNan by 0, unstack on 0D, slice with number begin/size, comparison true & false branches, logicalXor
    const tSmall = new TfTensor([1], 'float32', [1]);
    const tLarge = new TfTensor([2], 'float32', [1, 2]);
    expect(add(tSmall, tLarge).nData).toEqual([2, 3]);
    expect(equal(tSmall, tLarge).nData).toEqual([1, 0]);
    expect(equal(tLarge, tSmall).nData).toEqual([1, 0]);
    expect(notEqual(tSmall, tLarge).nData).toEqual([0, 1]);
    expect(less(tLarge, tSmall).nData).toEqual([0, 0]);
    expect(lessEqual(tLarge, tSmall).nData).toEqual([1, 0]);
    expect(greater(tSmall, tLarge).nData).toEqual([0, 0]);
    expect(greater(tLarge, tSmall).nData).toEqual([0, 1]);
    expect(greaterEqual(tSmall, tLarge).nData).toEqual([1, 0]);
    expect(
      logicalXor(new TfTensor([2], 'bool', [0, 1]), new TfTensor([2], 'bool', [0, 1])).nData,
    ).toEqual([0, 0]);
    expect(
      logicalXor(new TfTensor([2], 'bool', [1, 0]), new TfTensor([2], 'bool', [0, 1])).nData,
    ).toEqual([1, 1]);
    expect(divNoNan(tSmall, new TfTensor([1], 'float32', [0])).nData).toEqual([0]);
    expect(unstack(new TfTensor([], 'float32', [2]))).toHaveLength(1);
    expect(slice(new TfTensor([4], 'float32', [1, 2, 3, 4]), 1, 2).nData).toEqual([2, 3]);
    expect(slice(new TfTensor([4], 'float32', [1, 2, 3, 4]), 1).nData).toEqual([2, 3, 4]);

    // ops/nn.ts: both scalar and array strides/dilations/filterSize, 4D vs 5D in maxPool3d/avgPool3d, pool with array strides vs undefined strides, leakyRelu/elu/selu negative & positive
    const x4d = new TfTensor([1, 2, 2, 1], 'float32', [-1, 2, 3, 4]);
    const f4d = new TfTensor([1, 1, 1, 1], 'float32', [1]);
    expect(conv2d(x4d, f4d, 1, 'valid', 'NHWC', 1).shape).toEqual([1, 2, 2, 1]);
    expect(conv2d(x4d, f4d, [1, 1], 'valid', 'NHWC', [1, 1]).shape).toEqual([1, 2, 2, 1]);
    expect(depthwiseConv2d(x4d, f4d, 1, 'valid', 'NHWC', 1).shape).toEqual([1, 2, 2, 1]);
    expect(depthwiseConv2d(x4d, f4d, [1, 1], 'valid', 'NHWC', [1, 1]).shape).toEqual([1, 2, 2, 1]);
    expect(conv2dTranspose(x4d, f4d, [1, 2, 2, 1], 1, 'valid').shape).toEqual([1, 2, 2, 1]);
    expect(conv2dTranspose(x4d, f4d, [1, 2, 2, 1], [1, 1], 'valid').shape).toEqual([1, 2, 2, 1]);
    expect(maxPool(x4d, 1, 1, 'valid').shape).toEqual([1, 2, 2, 1]);
    const x5d = new TfTensor([1, 2, 2, 2, 1], 'float32', new Array(8).fill(1));
    const f5d = new TfTensor([1, 1, 1, 1, 1], 'float32', [1]);
    expect(conv3d(x5d, f5d, 1, 'valid').shape).toEqual([1, 2, 2, 2, 1]);
    expect(conv3d(x5d, f5d, [1, 1, 1], 'valid').shape).toEqual([1, 2, 2, 2, 1]);
    expect(conv3dTranspose(x5d, f5d, [1, 2, 2, 2, 1], [1, 1, 1], 'valid').shape).toEqual([
      1, 2, 2, 2, 1,
    ]);
    expect(maxPool3d(x5d, [1, 1, 1], [1, 1, 1], 'valid').shape).toEqual([1, 2, 2, 2, 1]);
    expect(maxPool3d(x4d, 1, 0, 'valid').shape).toEqual([1, 2, 2, 1]);
    expect(avgPool3d(x5d, [1, 1, 1], [1, 1, 1], 'valid').shape).toEqual([1, 2, 2, 2, 1]);
    expect(avgPool3d(x4d, 1, 0, 'valid').shape).toEqual([1, 2, 2, 1]);
    expect(pool(x4d, 1, 'max', 'valid').shape).toEqual([1, 2, 2, 1]);
    expect(pool(x4d, [1, 1], 'avg', 'valid', 1, [1, 1]).shape).toEqual([1, 2, 2, 1]);
    expect(leakyRelu(x4d).nData[0]).toBeLessThan(0);
    expect(elu(x4d).nData[0]).toBeLessThan(0);
    expect(selu(x4d).nData[0]).toBeLessThan(0);

    // ops/reduction.ts: keepDims=true and keepDims=false both with axis=undefined and axis=0
    const rT = new TfTensor([2, 2], 'float32', [1, 2, 3, 4]);
    for (const fn of [min, max, mean, prod, sum, all, any, logSumExp]) {
      expect(fn(rT, undefined, true).shape).toEqual([1, 1]);
      expect(fn(rT, undefined, false).shape).toEqual([]);
      expect(fn(rT, 0, true).shape).toEqual([1, 1]);
      expect(fn(rT, 0, false).shape).toEqual([1]);
    }
  });

  it('covers @onnx9000/serve remaining branches', async () => {
    const debugSpy = vi.spyOn(console, 'debug').mockImplementation(() => {});
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});

    // batcher.ts: non-Error rejection, and mixed sequence + non-sequence payload in batch
    const batcher = new DynamicBatcher(
      async () => {
        throw 'string batch error';
      },
      { maxBatchSize: 1, batchTimeoutMs: 10 },
    );
    await expect(batcher.add({ input_ids: [1, 2] })).rejects.toThrow('string batch error');

    const batcherPad = new DynamicBatcher(async (_batch, inputs) => [inputs, inputs], {
      maxBatchSize: 2,
      batchTimeoutMs: 10,
    });
    const [res1] = await Promise.all([
      batcherPad.add({ input_ids: [1, 2, 3] }),
      batcherPad.add({}),
    ]);
    expect(res1.input_ids).toHaveLength(2);

    // ensemble.ts: link without '.' in src
    const ens = new ModelEnsemble({
      name: 'test_ens',
      inputs: ['raw_in'],
      outputs: { out: 'n1.out' },
      nodes: [
        {
          id: 'n1',
          type: 'logic',
          inputs: { x: 'raw_in' },
          logic: async (inp) => inp.x,
        },
      ],
    });
    await ens.execute({ raw_in: 42 });

    // kserve.ts: binary & JSON request without id, INT32 input without data, and non-Error throw
    const srv = createServer();
    const binHeader = new TextEncoder().encode(
      JSON.stringify({
        inputs: [{ name: 'in0', shape: [1], datatype: 'FP32' }],
      }),
    );
    const binRes = await srv.fetch(
      new Request('http://localhost/v2/models/m1/infer', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/octet-stream',
          'Inference-Header-Content-Length': String(binHeader.byteLength),
        },
        body: binHeader,
      }),
    );
    expect(binRes.status).toBe(200);
    const kRes = await srv.fetch(
      new Request('http://localhost/v2/models/m1/infer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          inputs: [{ name: 'in0', shape: [1], datatype: 'INT32' }],
        }),
      }),
    );
    expect(kRes.status).toBe(200);

    const origText = Request.prototype.text;
    Request.prototype.text = () => Promise.reject('non-error text failure');
    try {
      const kErr = await srv.fetch(
        new Request('http://localhost/v2/models/m1/infer', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: '{}',
        }),
      );
      expect(kErr.status).toBe(400);
    } finally {
      Request.prototype.text = origText;
    }

    // lambda.ts: empty event, requestContext.http.method + rawPath, getRemainingTimeInMillis timeout, and non-Error rejection
    const lambdaHandler = createLambdaHandler(srv);
    const lRes1 = await lambdaHandler({}, {});
    expect(lRes1.statusCode).toBe(404);
    const lRes2 = await lambdaHandler(
      { requestContext: { http: { method: 'GET' } }, rawPath: '/v2/health/ready' },
      { getRemainingTimeInMillis: () => 5000 },
    );
    expect(lRes2.statusCode).toBe(200);

    const slowSrv = {
      fetch: () =>
        new Promise<Response>((resolve) => setTimeout(() => resolve(new Response()), 50)),
    } as never;
    const timeoutRes = await createLambdaHandler(slowSrv)(
      {},
      { getRemainingTimeInMillis: () => 10 },
    );
    expect(timeoutRes.statusCode).toBe(504);

    const strThrowSrv = {
      fetch: () => Promise.reject('lambda string error'),
    } as never;
    const strThrowRes = await createLambdaHandler(strThrowSrv)({}, {});
    expect(strThrowRes.statusCode).toBe(504);

    // logger.ts: circular object in JSON mode
    const jsonLogger = new Logger(LogLevel.DEBUG, true);
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    jsonLogger.info('circular test', circular);

    // router.ts: route handler throwing non-Error string
    const r = new Router();
    r.add('GET', '/throw-str', async () => {
      throw 'route string error';
    });
    const rRes = await r.handle(new Request('http://localhost/throw-str'));
    expect(rRes.status).toBe(500);

    // openai.ts: chat without messages, completions without model, embeddings without encoding_format/model and with btoa undefined
    await srv.fetch(
      new Request('http://localhost/v1/chat/completions', {
        method: 'POST',
        body: JSON.stringify({}),
      }),
    );
    await srv.fetch(
      new Request('http://localhost/v1/completions', {
        method: 'POST',
        body: JSON.stringify({ prompt: 'hi' }),
      }),
    );
    const origBtoa = globalThis.btoa;
    // @ts-expect-error testing Node fallback when btoa is undefined
    delete globalThis.btoa;
    try {
      await srv.fetch(
        new Request('http://localhost/v1/embeddings', {
          method: 'POST',
          body: JSON.stringify({ input: 'hi', encoding_format: 'base64' }),
        }),
      );
      await srv.fetch(
        new Request('http://localhost/v1/embeddings', {
          method: 'POST',
          body: JSON.stringify({ input: 'hi' }),
        }),
      );
    } finally {
      globalThis.btoa = origBtoa;
    }

    // node.ts: HTTP/2 server + handler with missing host/url/method and non-Error rejection
    const h2Server = serveNode(srv, 0, true);
    await new Promise<void>((resolve) => h2Server.once('listening', () => resolve()));
    await new Promise<void>((resolve) => h2Server.close(() => resolve()));

    const badSrv = {
      fetch: () => Promise.reject('node string error'),
    } as never;
    const h1Bad = serveNode(badSrv, 0, false);
    const reqListener = h1Bad.listeners('request')[0] as (
      req: unknown,
      res: unknown,
    ) => Promise<void>;
    const mockReq = {
      socket: { encrypted: false },
      headers: {},
      url: undefined,
      method: undefined,
      async *[Symbol.asyncIterator]() {},
    };
    const mockRes = {
      statusCode: 200,
      setHeader: vi.fn(),
      end: vi.fn(),
    };
    await reqListener(mockReq, mockRes);
    expect(mockRes.statusCode).toBe(500);
    await new Promise<void>((resolve) => h1Bad.close(() => resolve()));

    debugSpy.mockRestore();
    errSpy.mockRestore();
    logSpy.mockRestore();
  });
});
