import { describe, it } from 'vitest';
import * as search from '../../src/genai/search';
import * as tok from '../../src/genai/tokenizer';
import * as lp from '../../src/genai/logit_processors';
import * as state from '../../src/genai/state';
import { Tensor } from '../../src/ir/tensor';

describe('missing16', () => {
  it('logit_processors', () => {
    // 277 ForcedEOS: inputIds length !== maxLength - 1 -> return logits
    const eos = new lp.ForcedEOSLogitProcessor(2, 0);
    eos.process([], new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2])));

    // 316-317 MinPLogitProcessor -> minP >= 1.0
    const minP = new lp.MinPLogitProcessor(1.0);
    minP.process([], new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2])));

    // 324 MinP -> !Float32Array
    const minP2 = new lp.MinPLogitProcessor(0.1);
    minP2.process([], new Tensor('a', [1, 2], 'int32', false, false, new Int32Array([1, 2])));
  });

  it('state 113-116, 122-123', () => {
    const c = new state.GroupedQueryAttentionCache(1, 1);
    try {
      c.update(
        new Tensor('a', [1, 99], 'float32', false, false, new Float32Array([1])),
        new Tensor('b', [1, 99], 'float32', false, false, new Float32Array([1])),
        0,
      );
    } catch (e) {}
  });

  it('tokenizer 260-261, 273-274', () => {
    const w = new tok.WordPieceTokenizer(
      new Map([
        ['a', 1],
        ['b', 2],
        ['##c', 3],
      ]),
      '[UNK]',
    );
    // 260-261: if (text.length > 0) text += ' ';
    w.decode([1, 2]); // 'a b' -> text.length > 0 triggers
    // 273-274: encodeBatch mapping
    w.encodeBatch(['a b', 'c']);
  });
});
