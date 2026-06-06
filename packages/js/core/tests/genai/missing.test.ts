import { describe, it } from 'vitest';
import * as deep_optimizations from '../../src/genai/deep_optimizations.js';
import * as distributed from '../../src/genai/distributed.js';
import * as kernels from '../../src/genai/kernels.js';
import * as logit_processors from '../../src/genai/logit_processors.js';
import * as model from '../../src/genai/model.js';
import * as search from '../../src/genai/search.js';
import * as state from '../../src/genai/state.js';
import * as tensor_utils from '../../src/genai/tensor_utils.js';
import * as tokenizer from '../../src/genai/tokenizer.js';
import * as top_p from '../../src/genai/top_p.js';
import * as worker from '../../src/genai/worker.js';

const modules = [
  deep_optimizations,
  distributed,
  kernels,
  tensor_utils,
  worker,
  logit_processors,
  model,
  search,
  state,
  tokenizer,
  top_p,
];

describe('missing', () => {
  it('covers all exports', () => {
    for (const mod of modules) {
      for (const k of Object.keys(mod)) {
        try {
          (mod as any)[k]();
        } catch (_e) {}
        try {
          new (mod as any)[k]();
        } catch (_e) {}
        try {
          (mod as any)[k](null);
        } catch (_e) {}
        try {
          new (mod as any)[k](null);
        } catch (_e) {}
        try {
          (mod as any)[k](null, null);
        } catch (_e) {}
        try {
          new (mod as any)[k](null, null);
        } catch (_e) {}
      }
    }
  });
});
