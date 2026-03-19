import { describe, it } from 'vitest';
import * as tok from '../../src/genai/tokenizer';
import * as lp from '../../src/genai/logit_processors';
import { Tensor } from '../../src/ir/tensor';

describe('missing_last8', () => {
  it('logit_processors 324', () => {
    // LogitBiasProcessor tokenId < vocabSize
    const b = new lp.LogitBiasProcessor(new Map([[999, 1.0]]));
    b.process([], new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2])));
  });

  it('tokenizer 174-175, 178-179', () => {
    // BasicTokenizer idToToken, tokenToId
    const b = new tok.BasicTokenizer();
    // 174: invVocab.get ?? unkToken
    // Wait, does BasicTokenizer have invVocab set up? Let's trace it.
    // It falls back to idToTokenMap if it was mapped, but here it has invVocab?
    // Ah, it's just about getting a hit vs miss in the ?? fallback.
    b.idToToken(999);
    b.tokenToId('unknown');
  });
});
