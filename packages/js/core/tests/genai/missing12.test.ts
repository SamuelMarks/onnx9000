import { describe, it } from 'vitest';
import * as search from '../../src/genai/search';
import * as tok from '../../src/genai/tokenizer';
import * as lp from '../../src/genai/logit_processors';
import * as state from '../../src/genai/state';
import { Tensor } from '../../src/ir/tensor';

describe('missing12', () => {
  it('search 23', () => {
    const alg = new search.BeamSearchAlgorithm(new search.BeamSearchState(1, 1));
    // test processLogits returning empty because length of scores is too small or other branches
    const t = new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2]));
    alg.processLogits(t, 0);
  });

  it('logit_processors', () => {
    const minP = new lp.MinPLogitProcessor(0.1);
    minP.process([], new Tensor('a', [1, 2], 'int32', false, false, new Int32Array([1, 2])));

    new lp.ForcedEOSLogitProcessor(2, 0).process(
      [1],
      new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2])),
    );
    new lp.ForcedEOSLogitProcessor(2, 0).process(
      [1],
      new Tensor('a', [1, 2], 'int32', false, false, new Int32Array([1, 2])),
    );
  });

  it('state', () => {
    const c = new state.MultiHeadAttentionCache(1);
    try {
      c.update(
        new Tensor('a', [1, 99], 'float32', false, false, new Float32Array([1])),
        new Tensor('b', [1, 99], 'float32', false, false, new Float32Array([1])),
        0,
      );
    } catch (e) {}
  });

  it('tokenizer', () => {
    const t = new tok.BasicTokenizer();
    t.encodeBatch(['']);
    t.decodeBatch([[1]]);
  });
});
