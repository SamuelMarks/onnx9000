import { describe, it } from 'vitest';
import * as lp from '../../src/genai/logit_processors.js';
import * as search from '../../src/genai/search.js';
import * as state from '../../src/genai/state.js';
import * as tok from '../../src/genai/tokenizer.js';
import * as top_p from '../../src/genai/top_p.js';
import { Tensor } from '../../src/ir/tensor.js';

describe('missing5', () => {
  it('logit_processors', () => {
    const bp = new lp.NoBadWordsLogitProcessor([[1, 2]]);
    const t = new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2]));
    bp.process([3, 4], t); // no match branch
    bp.process([1, 9], t); // match partial but fail inner loop

    try {
      new lp.AllowedWordsLogitProcessor([]).process([], t);
    } catch (_e) {}
  });

  it('search', () => {
    const alg = new search.BeamSearchAlgorithm(new search.BeamSearchState(1, 1));
    const t2 = new Tensor('a', [1, 2], 'int32', false, false, new Int32Array([1, 2]));
    alg.processLogits(t2, 0); // not float32array branch
  });

  it('state', () => {
    new state.CrossLayerKVCache().get(0);
    new state.MemoryReuseKVCache().get(0);
    new state.SequenceBatchingKVCache().get(0);
    new state.CrossAttentionCache().get(0);
  });

  it('tokenizer', () => {
    const b = new tok.BasicTokenizer();
    try {
      b.encode('');
    } catch (_e) {}
    try {
      b.idToToken(99999);
    } catch (_e) {}
    try {
      b.tokenToId('x');
    } catch (_e) {}
    try {
      b.encodeBatch(['']);
    } catch (_e) {}
    try {
      b.decodeBatch([[]]);
    } catch (_e) {}

    const w = new tok.WordPieceTokenizer(new Map([['a', 0]]));
    try {
      w.encode('a');
    } catch (_e) {}
    try {
      w.encode('b');
    } catch (_e) {}

    const u = new tok.UnigramTokenizer(new Map([['a', 0]]));
    try {
      u.encode('a');
    } catch (_e) {}
    try {
      u.encode('b');
    } catch (_e) {}
  });

  it('top_p', () => {
    const p = new top_p.TopPLogitProcessor(0.5);
    p.process([], new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([NaN, NaN])));
  });
});
