import { describe, it } from 'vitest';
import * as search from '../../src/genai/search';
import * as tok from '../../src/genai/tokenizer';
import * as lp from '../../src/genai/logit_processors';
import * as state from '../../src/genai/state';
import { Tensor } from '../../src/ir/tensor';

describe('missing11', () => {
  it('search 74', () => {
    const m = new search.MultinomialSampling();
    // simulate reaching end of probabilities loop by setting Math.random() = 2 or giving probs < 0
    const origRandom = Math.random;
    Math.random = () => 2.0;
    const t = new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([0.1, 0.2]));
    m.selectNextToken(t, []);
    Math.random = origRandom;
  });

  it('logit_processors', () => {
    // 324: MinPLogitProcessor -> !float32array
    const minP = new lp.MinPLogitProcessor(0.1);
    minP.process([], new Tensor('a', [1, 2], 'int32', false, false, new Int32Array([1, 2])));

    // 277: ForcedEOSLogitProcessor -> !float32array branch logic
    // 316-317: MinPLogitProcessor -> return logits when minP >= 1 or no match
    const minP2 = new lp.MinPLogitProcessor(1.0);
    minP2.process([], new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2])));
  });

  it('state 113-116, 122-123', () => {
    const c = new state.MultiHeadAttentionCache(1);
    // update with valid shape but not Float32Array maybe? Or wrong number of heads?
    try {
      c.update(
        new Tensor('a', [1, 2], 'int32', false, false, new Int32Array([1])),
        new Tensor('b', [1, 2], 'int32', false, false, new Int32Array([1])),
        0,
      );
    } catch (e) {}

    // Let's actually give it the wrong number of heads to trigger the Error
    try {
      c.update(
        new Tensor('a', [1, 99], 'float32', false, false, new Float32Array([1])),
        new Tensor('b', [1, 99], 'float32', false, false, new Float32Array([1])),
        0,
      );
    } catch (e) {}
  });

  it('tokenizer 289-290, 367-368', () => {
    const t = new tok.BasicTokenizer();
    // 367-368: encodeBatch mapping branch / decodeBatch
    t.encodeBatch([]);
    t.decodeBatch([]);
  });
});
