import { describe, it } from 'vitest';
import * as tok from '../../src/genai/tokenizer';
import * as lp from '../../src/genai/logit_processors';
import { Tensor } from '../../src/ir/tensor';

describe('missing_last9', () => {
  it('logit_processors', () => {
    const b = new lp.LogitBiasProcessor(new Map([[0, 1.0]]));
    const t = new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2]));
    b.process([], t);

    // 277 ForcedEOS returning logits: inputIds length !== maxLength - 1 OR !Float32Array
    new lp.ForcedEOSLogitProcessor(5, 0).process([1], t); // inputIds.length = 1, maxLength - 1 = 4. Miss.
    new lp.ForcedEOSLogitProcessor(2, 0).process(
      [1],
      new Tensor('a', [1, 2], 'int32', false, false, new Int32Array([1, 2])),
    ); // not float32. Miss.

    // 316-317 LogitBias returning logits: map size === 0 OR !Float32Array
    new lp.LogitBiasProcessor(new Map()).process([], t); // size === 0. Miss.
    new lp.LogitBiasProcessor(new Map([[0, 1.0]])).process(
      [],
      new Tensor('a', [1, 2], 'int32', false, false, new Int32Array([1, 2])),
    ); // not float32. Miss.
  });

  it('tokenizer 174-175, 178-179', () => {
    // BasicTokenizer fallback for idToToken (?? this.unkToken) and tokenToId (?? this.unkTokenId)
    const b = new tok.BasicTokenizer();
    // Since invVocab / vocab are missing, it defaults
    b.idToToken(999);
    b.tokenToId('UNKNOWN');
  });
});
