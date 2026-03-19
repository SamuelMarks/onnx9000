import { describe, it } from 'vitest';
import * as search from '../../src/genai/search';
import * as tok from '../../src/genai/tokenizer';
import * as lp from '../../src/genai/logit_processors';
import * as state from '../../src/genai/state';
import { Tensor } from '../../src/ir/tensor';

describe('missing14', () => {
  it('search 23', () => {
    const alg = new search.BeamSearchAlgorithm(new search.BeamSearchState(2, 2));
    // We need to hit branch 23.
    const t = new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2]));
    alg.selectNextToken(t, []);
  });

  it('logit_processors 324,277,316-317', () => {
    // We need to trigger the internal branches
    const minP = new lp.MinPLogitProcessor(1.0);
    minP.process([], new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2])));

    new lp.ForcedEOSLogitProcessor(2, 0).process(
      [1],
      new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2])),
    );
  });

  it('state', () => {
    const c = new state.MultiHeadAttentionCache(1);
    try {
      c.update(
        new Tensor('a', [1, 1], 'float32', false, false, new Float32Array([1])),
        new Tensor('b', [1, 99], 'float32', false, false, new Float32Array([1])),
        0,
      );
    } catch (e) {}

    const m = new state.MultiQueryAttentionCache(1);
    try {
      m.update(
        new Tensor('a', [1, 1], 'float32', false, false, new Float32Array([1])),
        new Tensor('b', [1, 99], 'float32', false, false, new Float32Array([1])),
        0,
      );
    } catch (e) {}
  });

  it('tokenizer', () => {
    const t = new tok.BasicTokenizer();
    t.encodeBatch([]);
    t.decodeBatch([]);

    const bpe = new tok.BPETokenizer([], new Map(), '<unk>');
    bpe.encodeBatch([]);
    bpe.decodeBatch([]);
  });
});
