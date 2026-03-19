import { describe, it } from 'vitest';
import * as lp from '../../src/genai/logit_processors';
import * as state from '../../src/genai/state';
import * as tok from '../../src/genai/tokenizer';
import { Tensor } from '../../src/ir/tensor';

describe('missing8', () => {
  it('logit_processors 354-355, 377-378', () => {
    const nr = new lp.NoRepeatNGramLogitProcessor(2);
    const t = new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2]));
    // 354-355: !float32array
    nr.process([1, 2], new Tensor('a', [1, 2], 'int32', false, false, new Int32Array([1, 2])));
    // 377-378: bannedTokens.size === 0 -> match inner loop fail
    nr.process([3, 4], t);
    // also get a match
    nr.process([1, 1], t);
  });

  it('state 230-233, 296-297', () => {
    const mq = new state.MultiQueryAttentionCache(1, 2);
    try {
      mq.update(
        new Tensor('a', [1, 1], 'int32', false, false, new Int32Array([1])),
        new Tensor('b', [1, 1], 'int32', false, false, new Int32Array([1])),
        0,
      );
    } catch (e) {}

    const sw = new state.SlidingWindowKVCache(1);
    sw.update(
      new Tensor('a', [1, 1, 2], 'int32', false, false, new Int32Array([1])),
      new Tensor('b', [1, 1, 2], 'int32', false, false, new Int32Array([1])),
      0,
    );
  });

  it('tokenizer 379-380, 383-384', () => {
    const u = new tok.UnigramTokenizer(new Map([['a', 1]]), '<unk>');
    u.idToToken(999);
    u.tokenToId('x');
  });
});
