import { expect, test } from 'vitest';
import { Tensor } from '../src/ir/tensor.js';
import { resnet18 } from '../src/models/resnet.js';
import { efficientnetB0 } from '../src/models/efficientnet.js';
import { convnextTiny } from '../src/models/convnext.js';
import { mobilevitS } from '../src/models/mobilevit.js';

test('ResNet', () => {
  const model = resnet18();
  const x = new Tensor('x', [1, 3, 224, 224], 1, false, false, new Float32Array());
  const out = model.call(x);
  expect(out).toBeInstanceOf(Tensor);
});

test('EfficientNet', () => {
  const model = efficientnetB0();
  const x = new Tensor('x', [1, 3, 224, 224], 1, false, false, new Float32Array());
  const out = model.call(x);
  expect(out).toBeInstanceOf(Tensor);
});

test('ConvNeXt', () => {
  const model = convnextTiny();
  const x = new Tensor('x', [1, 3, 224, 224], 1, false, false, new Float32Array());
  const out = model.call(x);
  expect(out).toBeInstanceOf(Tensor);
});

test('MobileViT', () => {
  const model = mobilevitS();
  const x = new Tensor('x', [1, 3, 224, 224], 1, false, false, new Float32Array());
  const out = model.call(x);
  expect(out).toBeInstanceOf(Tensor);
});

import { vitBasePatch16_224 } from '../src/models/vit.js';
import { swinT } from '../src/models/swin.js';
import { maeVitBasePatch16 } from '../src/models/mae.js';

test('ViT', () => {
  const model = vitBasePatch16_224();
  const x = new Tensor('x', [1, 3, 224, 224], 1, false, false, new Float32Array());
  const out = model.call(x);
  expect(out).toBeInstanceOf(Tensor);
});

test('Swin', () => {
  const model = swinT();
  const x = new Tensor('x', [1, 3, 224, 224], 1, false, false, new Float32Array());
  const out = model.call(x);
  expect(out).toBeInstanceOf(Tensor);
});

test('MAE', () => {
  const model = maeVitBasePatch16();
  const x = new Tensor('x', [1, 3, 224, 224], 1, false, false, new Float32Array());
  const maskIndices = new Tensor('mask_indices', [1, 49], 7, false, false, new BigInt64Array());
  const out = model.call(x, maskIndices);
  expect(out).toBeInstanceOf(Tensor);
});

import { llama7b } from '../src/models/llama.js';
import { mixtral8x7b } from '../src/models/mixtral.js';
import { mamba130m } from '../src/models/mamba.js';
import { rwkvV4 } from '../src/models/rwkv.js';

test('LLaMA', () => {
  const model = llama7b();
  const x = new Tensor('x', [1, 32], 7, false, false, new BigInt64Array());
  const pos = new Tensor('pos', [1, 32], 7, false, false, new BigInt64Array());
  const out = model.call(x, pos);
  expect(out).toBeInstanceOf(Tensor);
});

test('Mixtral', () => {
  const model = mixtral8x7b();
  const x = new Tensor('x', [1, 32], 7, false, false, new BigInt64Array());
  const pos = new Tensor('pos', [1, 32], 7, false, false, new BigInt64Array());
  const out = model.call(x, pos);
  expect(out).toBeInstanceOf(Tensor);
});

test('Mamba', () => {
  const model = mamba130m();
  const x = new Tensor('x', [1, 32], 7, false, false, new BigInt64Array());
  const out = model.call(x);
  expect(out).toBeInstanceOf(Tensor);
});

test('RWKV', () => {
  const model = rwkvV4();
  const x = new Tensor('x', [1, 32], 7, false, false, new BigInt64Array());
  const out = model.call(x);
  expect(out).toBeInstanceOf(Tensor);
});

import { whisperTiny } from '../src/models/whisper.js';
import { ditXl2 } from '../src/models/dit.js';
import { clipVitBasePatch16 } from '../src/models/clip.js';

test('Whisper', () => {
  const model = whisperTiny();
  const x = new Tensor('x', [1, 80, 3000], 1, false, false, new Float32Array());
  const decIds = new Tensor('dec_ids', [1, 32], 7, false, false, new BigInt64Array());
  const out = model.call(x, decIds);
  expect(out).toBeInstanceOf(Tensor);
});

test('DiT', () => {
  const model = ditXl2();
  const x = new Tensor('x', [1, 4, 32, 32], 1, false, false, new Float32Array());
  const t = new Tensor('t', [1, 1152], 1, false, false, new Float32Array());
  const out = model.call(x, t);
  expect(out).toBeInstanceOf(Tensor);
});

test('CLIP', () => {
  const model = clipVitBasePatch16();
  const img = new Tensor('img', [1, 3, 224, 224], 1, false, false, new Float32Array());
  const text = new Tensor('text', [1, 77], 7, false, false, new BigInt64Array());
  const [out1, out2] = model.call(img, text);
  expect(out1).toBeInstanceOf(Tensor);
  expect(out2).toBeInstanceOf(Tensor);
});
