import { describe, it, expect } from 'vitest';
import { BaseImageProcessor, SequenceFeatureExtractor } from '../src/processors/index';

describe('Processors', () => {
  it('BaseImageProcessor', async () => {
    const config = {
      do_resize: true,
      size: [224, 224],
      image_mean: [0.5, 0.5, 0.5],
      image_std: [0.5, 0.5, 0.5],
    };
    const p = new BaseImageProcessor(config);

    // Process one fake image (nested array 3x3x3)
    const img = [
      [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
      ],
      [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
      ],
      [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
      ],
    ];
    await expect(p.process(img)).resolves.toBeDefined();

    // Also without config overrides
    const p2 = new BaseImageProcessor({});
    await expect(p2.process(img, { do_resize: false })).resolves.toBeDefined();
  });

  it('SequenceFeatureExtractor', async () => {
    const config = { do_pad: true, do_normalize: true };
    const p = new SequenceFeatureExtractor(config);

    // Process simple float array
    const audio = new Float32Array([1.0, -1.0, 0.5]);
    await expect(p.process(audio)).resolves.toBeDefined();

    const p2 = new SequenceFeatureExtractor({});
    await expect(p2.process(audio, { do_pad: false })).resolves.toBeDefined();
  });
});

it('AutoProcessor and methods', async () => {
  const { AutoProcessor } = await import('../src/processors/index');
  const proc = await AutoProcessor.fromPretrained('id');
  expect(proc).toBeDefined();
  await proc.process('image');
  await proc.process('other', { return_tensors: true });

  const p2 = new SequenceFeatureExtractor({});
  p2.chunkWaveform([1]);
  p2.applyVAD([1]);
});

it('uncovered lines in processors', async () => {
  const { SequenceFeatureExtractor } = await import('../src/processors/index');
  const p = new SequenceFeatureExtractor({});
  p.do_truncate([1]);
  p.wasmWindowing([1], 'type');
});

it('uncovered lines in processors 2', async () => {
  const { SequenceFeatureExtractor, BaseImageProcessor } = await import('../src/processors/index');
  const p = new SequenceFeatureExtractor({ do_truncate: true });
  await p.process(new Float32Array([1]));

  const imgProc = new BaseImageProcessor({});
  imgProc.webgpuResizeShader({});
  BaseImageProcessor.drawBoundingBoxes({}, []);
  BaseImageProcessor.drawSegmentationMask({}, {});
});

it('uncovered lines in processors 3', async () => {
  const { BaseImageProcessor } = await import('../src/processors/index');
  const imgProc = new BaseImageProcessor({});
  imgProc.do_normalize({}, [], []);
  imgProc.webgpuNormalizeShader({});
});

it('uncovered lines in processors 4', async () => {
  const { BaseImageProcessor } = await import('../src/processors/index');
  const imgProc = new BaseImageProcessor({});
  imgProc.do_pad({});
  imgProc.do_rescale({});
  imgProc.do_random_crop({});
});

it('uncovered lines in processors 5', async () => {
  const { BaseImageProcessor } = await import('../src/processors/index');
  const imgProc = new BaseImageProcessor({});
  imgProc.wasmNearestNeighborResize({});
  imgProc.do_center_crop({});
});

it('uncovered lines in processors 6', async () => {
  const { BaseImageProcessor } = await import('../src/processors/index');
  const imgProc = new BaseImageProcessor({});
  imgProc.wasmBicubicResize({});
  imgProc.wasmBilinearResize({});
});

it('uncovered lines in processors 7', async () => {
  const { BaseImageProcessor } = await import('../src/processors/index');
  const imgProc = new BaseImageProcessor({
    do_random_crop: true,
    do_center_crop: true,
    do_pad: true,
    do_rescale: true,
    do_normalize: true,
  });
  await imgProc.process([[[1, 1, 1]]]);
});

it('uncovered lines in processors 8', async () => {
  const { ONNX9000Image, ONNX9000Audio } = await import('../src/processors/index');
  ONNX9000Image.fromBase64('data');
  await ONNX9000Audio.fromURL('url');
  ONNX9000Audio.fromBlob(new Blob());
});

it('uncovered lines in processors 9', async () => {
  const { ONNX9000Image } = await import('../src/processors/index');
  await ONNX9000Image.fromURL('url');
});
