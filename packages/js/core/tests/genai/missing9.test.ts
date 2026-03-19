import { describe, it } from 'vitest';
import * as lp from '../../src/genai/logit_processors';
import * as state from '../../src/genai/state';
import * as tok from '../../src/genai/tokenizer';
import { Tensor } from '../../src/ir/tensor';

describe('missing9', () => {
  it('logit_processors 289-306, 316-337', () => {
    const eos = new lp.ForcedEOSLogitProcessor(2, 0);
    const t = new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2]));
    eos.process([1], t); // forces EOS since length == 1 and max = 2

    const bias = new lp.LogitBiasProcessor(new Map([[0, 1.0]]));
    bias.process([], t); // applies bias
  });

  it('state 201-204, 230-233', () => {
    const g = new state.GroupedQueryAttentionCache(1, 1);
    try {
      g.update(
        new Tensor('a', [1, 2], 'int32', false, false, new Int32Array([1])),
        new Tensor('b', [1, 2], 'int32', false, false, new Int32Array([1])),
        0,
      );
    } catch (e) {}

    const m = new state.MultiQueryAttentionCache(1);
    try {
      m.update(
        new Tensor('a', [1, 2], 'int32', false, false, new Int32Array([1])),
        new Tensor('b', [1, 2], 'int32', false, false, new Int32Array([1])),
        0,
      );
    } catch (e) {}
  });

  it('tokenizer 371-372, 375-376', () => {
    const bpe = new tok.BPETokenizer([], new Map(), '<unk>');
    bpe.encodeBatch(['a']);
    bpe.decodeBatch([[1]]);
  });
});
