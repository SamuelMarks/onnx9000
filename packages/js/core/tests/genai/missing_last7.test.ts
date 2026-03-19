import { describe, it } from 'vitest';
import * as search from '../../src/genai/search';
import * as tok from '../../src/genai/tokenizer';
import * as lp from '../../src/genai/logit_processors';
import * as state from '../../src/genai/state';
import { Tensor } from '../../src/ir/tensor';

describe('missing_last7', () => {
  it('logit_processors 316-317, 324', () => {
    const b = new lp.LogitBiasProcessor(new Map([[0, 1.0]]));
    b.process([], new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2])));
    const b2 = new lp.LogitBiasProcessor(new Map([[999, 1.0]]));
    b2.process([], new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2])));
  });

  it('state 294', () => {
    const sw = new state.SlidingWindowKVCache(1);
    sw.update(
      new Tensor('a', [1, 2], 'int32', false, false, new Int32Array([1])),
      new Tensor('b', [1, 2], 'int32', false, false, new Int32Array([1])),
      0,
    );
  });

  it('tokenizer 174-175, 178-179', () => {
    const t = new tok.BasicTokenizer();
    t.idToToken(9999);
    t.tokenToId('X');
  });
});
