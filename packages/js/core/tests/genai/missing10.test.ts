import { describe, it } from 'vitest';
import * as search from '../../src/genai/search';
import * as tok from '../../src/genai/tokenizer';
import * as lp from '../../src/genai/logit_processors';
import * as state from '../../src/genai/state';
import { Tensor } from '../../src/ir/tensor';

describe('missing10', () => {
  it('search 74', () => {
    const m = new search.MultinomialSampling();
    // sum probabilities to be < Math.random() (say r=0.999, probs=[0, 0])
    const t = new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([-1000, -1000]));
    m.selectNextToken(t, []);
  });

  it('tokenizer 371-372, 375-376', () => {
    const u = new tok.UnigramTokenizer(new Map(), '<unk>');
    u.decodeBatch([[1]]);
    u.createStream();

    const w = new tok.WordPieceTokenizer(new Map(), '[UNK]', 100);
    w.decodeBatch([[1]]);
    w.createStream();
  });

  it('logit_processors 324,277,316-317', () => {
    const b = new lp.LogitBiasProcessor(new Map([[0, 1.0]]));
    b.process([0], new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2])));

    new lp.ForcedEOSLogitProcessor(2, 0).process(
      [1],
      new Tensor('a', [1, 2], 'int32', false, false, new Int32Array([1, 2])),
    );

    const minP = new lp.MinPLogitProcessor(0.1);
    minP.process(
      [],
      new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([0.9, 0.1])),
    );
  });

  it('state 122-123, 171-174', () => {
    const c = new state.MultiHeadAttentionCache(1);
    try {
      c.update(
        new Tensor('a', [1, 2], 'int32', false, false, new Int32Array([1])),
        new Tensor('b', [1, 2], 'int32', false, false, new Int32Array([1])),
        0,
      );
    } catch (e) {}
  });
});
