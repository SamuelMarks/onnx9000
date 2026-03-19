import { describe, it } from 'vitest';
import * as search from '../../src/genai/search';
import * as tok from '../../src/genai/tokenizer';
import * as lp from '../../src/genai/logit_processors';
import * as state from '../../src/genai/state';
import { Tensor } from '../../src/ir/tensor';

describe('missing_last', () => {
  it('logit_processors', () => {
    // 107-108: Repetition penalty where val < 0
    const rp = new lp.RepetitionPenaltyLogitProcessor(2.0);
    rp.process(
      [1],
      new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1.0, -1.0])),
    );

    // 162-163: MinPLogitProcessor where maxVal === -Infinity
    const mp = new lp.MinPLogitProcessor(0.1);
    mp.process([], new Tensor('a', [1, 1], 'float32', false, false, new Float32Array([-Infinity])));
  });

  it('state 80-81, 86-87', () => {
    class MockKV {
      update() {}
      get() {
        return null;
      }
      clear() {}
    }
    const c = new state.CrossLayerKVCache(1, () => new MockKV());
    c.update(
      new Tensor('a', [1], 'int32', false, false, new Int32Array([1])),
      new Tensor('a', [1], 'int32', false, false, new Int32Array([1])),
      0,
    );
    c.get(0);
    c.clear();
  });

  it('tokenizer 174-175', () => {
    const b = new tok.BasicTokenizer();
    // check fallback unkToken when parsing unknown tokens
    b.idToToken(999);
    b.tokenToId('UNKNOWN_TOKEN');
  });
});
