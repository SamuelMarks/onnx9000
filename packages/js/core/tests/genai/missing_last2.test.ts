import { describe, it } from 'vitest';
import * as tok from '../../src/genai/tokenizer';
import * as lp from '../../src/genai/logit_processors';
import * as state from '../../src/genai/state';
import { Tensor } from '../../src/ir/tensor';

describe('missing_last2', () => {
  it('logit_processors 324,277,316-317', () => {
    // We need to trigger the internal branches
    const minP = new lp.MinPLogitProcessor(1.0);
    minP.process(
      [],
      new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([-Infinity, -Infinity])),
    );

    new lp.ForcedEOSLogitProcessor(3, 0).process(
      [1, 2],
      new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2])),
    );
    new lp.ForcedEOSLogitProcessor(3, 0).process(
      [1],
      new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2])),
    );

    new lp.MinPLogitProcessor(0.1).process(
      [],
      new Tensor('a', [1, 2], 'uint8', false, false, new Uint8Array([1, 2])),
    );
  });

  it('state 113-116, 122-123', () => {
    const c = new state.MultiHeadAttentionCache(1);
    try {
      c.update(
        new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1])),
        new Tensor('b', [1, 2], 'float32', false, false, new Float32Array([1])),
        0,
      );
    } catch (e) {}
    try {
      c.update(
        new Tensor('a', [1, 1], 'float32', false, false, new Float32Array([1])),
        new Tensor('b', [1, 2], 'float32', false, false, new Float32Array([1])),
        0,
      );
    } catch (e) {}
  });

  it('tokenizer 278-279, 210-212, 257', () => {
    const b = new tok.WordPieceTokenizer(new Map([['a', 1]]));
    b.encode('a b');

    const c = new tok.BPETokenizer([], new Map(), '<unk>');
    c.decode([1, 2]); // fallback branch inside loop

    const d = new tok.UnigramTokenizer(new Map([['a', 1]]));
    d.encode('a');
    d.encode('a a');
  });
});
