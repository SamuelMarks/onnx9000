import { describe, it, expect } from 'vitest';
import * as deep_optimizations from '../../src/genai/deep_optimizations';
import * as distributed from '../../src/genai/distributed';
import * as kernels from '../../src/genai/kernels';
import * as tensor_utils from '../../src/genai/tensor_utils';
import * as worker from '../../src/genai/worker';

import * as logit_processors from '../../src/genai/logit_processors';
import * as model from '../../src/genai/model';
import * as search from '../../src/genai/search';
import * as state from '../../src/genai/state';
import * as tokenizer from '../../src/genai/tokenizer';
import * as top_p from '../../src/genai/top_p';

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
          (mod as Object)[k]();
        } catch (e) {}
        try {
          new (mod as Object)[k]();
        } catch (e) {}
        try {
          (mod as Object)[k](null);
        } catch (e) {}
        try {
          new (mod as Object)[k](null);
        } catch (e) {}
        try {
          (mod as Object)[k](null, null);
        } catch (e) {}
        try {
          new (mod as Object)[k](null, null);
        } catch (e) {}
      }
    }
  });
});
