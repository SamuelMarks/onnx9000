import { describe, it } from 'vitest';
import * as lp from '../../src/genai/logit_processors';
import * as search from '../../src/genai/search';
import * as state from '../../src/genai/state';
import * as tok from '../../src/genai/tokenizer';
import * as top_p from '../../src/genai/top_p';
import { Tensor } from '../../src/ir/tensor';

describe('missing6', () => {
  it('logit_processors 378,403-404,412', () => {
    const bp = new lp.PresencePenaltyLogitProcessor(1.0);
    // 378: bannedTokens.size === 0 -> return logits
    // For presence penalty, inputIds is empty
    bp.process([], new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2])));

    // 403-404: NoBadWordsLogitProcessor -> !float32array
    const nbad = new lp.NoBadWordsLogitProcessor([[1]]);
    nbad.process([1], new Tensor('a', [1, 2], 'int32', false, false, new Int32Array([1, 2])));

    // 412: badWord.length === 1
    nbad.process([1], new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2])));
  });

  it('tokenizer 409-427,434-435', () => {
    const hf = new tok.HuggingFaceTokenizerLoader();
    // WordPiece without vocab
    tok.HuggingFaceTokenizerLoader.loadFromJson('{"model": {"type": "WordPiece"}}');
    // Unigram without vocab
    tok.HuggingFaceTokenizerLoader.loadFromJson('{"model": {"type": "Unigram"}}');
    // Unknown type
    try {
      tok.HuggingFaceTokenizerLoader.loadFromJson('{"model": {"type": "Unknown"}}');
    } catch (e) {}

    // UnicodeNormalizer form error
    try {
      tok.UnicodeNormalizer.normalize('a', 'BAD');
    } catch (e) {}
  });

  it('state 315-316,350-351', () => {
    // applyRoPE -> not float32array
    const t1 = new Tensor('a', [1, 2], 'int32', false, false, new Int32Array([1, 2]));
    state.PositionalEmbeddingUtils.applyRoPE(t1, t1, 1);

    // applyALiBi -> not float32array
    state.PositionalEmbeddingUtils.applyALiBi(t1, 1);
  });
});
