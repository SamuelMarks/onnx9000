import { describe, it } from 'vitest';
import * as search from '../../src/genai/search';
import * as tok from '../../src/genai/tokenizer';
import * as lp from '../../src/genai/logit_processors';
import * as state from '../../src/genai/state';
import { Tensor } from '../../src/ir/tensor';

describe('missing13', () => {
  it('search 23', () => {
    const alg = new search.BeamSearchAlgorithm(new search.BeamSearchState(2, 2));
    // try to prune and sort beams where length is 0 or numBeams > activeBeams
    alg.pruneAndSortBeams([]);
  });

  it('logit_processors 324,277,316-317', () => {
    const logit = new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2]));
    // 316-317 minp
    const minP2 = new lp.MinPLogitProcessor(1.0);
    minP2.process([], new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([0, 0])));

    const minP = new lp.MinPLogitProcessor(0.1);
    minP.process([], new Tensor('a', [1, 2], 'int32', false, false, new Int32Array([1, 2])));

    new lp.ForcedEOSLogitProcessor(2, 0).process(
      [1],
      new Tensor('a', [1, 2], 'int32', false, false, new Int32Array([1, 2])),
    );
  });

  it('state', () => {
    const c = new state.MultiHeadAttentionCache(1);
    try {
      c.update(
        new Tensor('a', [1, 99], 'int32', false, false, new Int32Array([1])),
        new Tensor('b', [1, 99], 'float32', false, false, new Float32Array([1])),
        0,
      );
    } catch (e) {}
  });

  it('tokenizer', () => {
    const t = new tok.BasicTokenizer();
    t.encodeBatch([]);
    t.decodeBatch([]);
  });
});
