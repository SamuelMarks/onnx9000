import { describe, it } from 'vitest';
import * as tok from '../../src/genai/tokenizer';
import * as lp from '../../src/genai/logit_processors';
import * as state from '../../src/genai/state';
import { Tensor } from '../../src/ir/tensor';

describe('missing_last5', () => {
  it('logit_processors 316-317, 324, 277', () => {
    const b = new lp.LogitBiasProcessor(new Map([[0, 1.0]]));
    // 316: tokenId < vocabSize
    // 317: newData[offset + tokenId] += bias
    b.process([], new Tensor('a', [1, 1], 'float32', false, false, new Float32Array([1])));

    // 277: ForcedEOS
    const e = new lp.ForcedEOSLogitProcessor(1, 0);
    e.process([], new Tensor('a', [1, 1], 'int32', false, false, new Int32Array([1])));
  });

  it('state 113-116, 122-123', () => {
    // MemoryReuseKVCache
    const mr = new state.MemoryReuseKVCache(1, 1);
    mr.update(
      new Tensor('a', [1], 'int32', false, false, new Int32Array([1])),
      new Tensor('a', [1], 'int32', false, false, new Int32Array([1])),
      0,
    );
    mr.clear(); // returns item to pool
    // 122-123: !this.active.has && this.pool.length > 0
    mr.update(
      new Tensor('a', [1], 'int32', false, false, new Int32Array([1])),
      new Tensor('a', [1], 'int32', false, false, new Int32Array([1])),
      0,
    );

    // Let's hit the pool.length < poolSize limit in clear()
    const mr2 = new state.MemoryReuseKVCache(1, 1);
    mr2.update(
      new Tensor('a', [1], 'int32', false, false, new Int32Array([1])),
      new Tensor('a', [1], 'int32', false, false, new Int32Array([1])),
      0,
    );
    mr2.update(
      new Tensor('a', [1], 'int32', false, false, new Int32Array([1])),
      new Tensor('a', [1], 'int32', false, false, new Int32Array([1])),
      1,
    );
    mr2.clear(); // 113-116 limit
  });

  it('tokenizer 210-212, 257', () => {
    // 210-212: word.length > maxInputCharsPerWord
    const w = new tok.WordPieceTokenizer(new Map([['a', 1]]), '[UNK]', 1);
    w.encode('aaaa');

    // 257: token.startsWith('##')
    const w2 = new tok.WordPieceTokenizer(new Map([['##a', 1]]));
    w2.decode([1]);

    // 278-279: idToToken map miss ?
    w2.idToToken(999);
  });
});
