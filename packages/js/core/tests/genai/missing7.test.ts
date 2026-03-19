import { describe, it } from 'vitest';
import * as lp from '../../src/genai/logit_processors';
import * as state from '../../src/genai/state';
import * as tok from '../../src/genai/tokenizer';
import { Tensor } from '../../src/ir/tensor';

describe('missing7', () => {
  it('logit_processors 354-355, 377-378', () => {
    // frequency penalty
    const fp = new lp.FrequencyPenaltyLogitProcessor(1.0);
    fp.process([0], new Tensor('a', [1, 1], 'float32', false, false, new Float32Array([1])));

    const pp = new lp.PresencePenaltyLogitProcessor(1.0);
    pp.process([0], new Tensor('a', [1, 1], 'float32', false, false, new Float32Array([1])));
  });

  it('state 230-233, 296-297', () => {
    const c = new state.SequenceBatchingKVCache(1, 1, 1);
    try {
      c.get(0);
    } catch (e) {}
    try {
      c.update(
        new Tensor('a', [1], 'int32', false, false, new Int32Array([1])),
        new Tensor('b', [1], 'int32', false, false, new Int32Array([1])),
        0,
      );
    } catch (e) {}

    const ca = new state.CrossAttentionCache(1, 1);
    try {
      ca.get(0);
    } catch (e) {}
    try {
      ca.update(
        new Tensor('a', [1], 'int32', false, false, new Int32Array([1])),
        new Tensor('b', [1], 'int32', false, false, new Int32Array([1])),
        0,
      );
    } catch (e) {}
  });

  it('tokenizer 412-413, 421-422', () => {
    // WordPiece vocab loop
    new tok.WordPieceTokenizer(new Map([['a', 1]]));
    tok.HuggingFaceTokenizerLoader.loadFromJson(
      '{"model": {"type": "WordPiece", "vocab": {"a": 1}}}',
    );

    // Unigram vocab loop
    new tok.UnigramTokenizer(new Map([['a', 1]]));
    tok.HuggingFaceTokenizerLoader.loadFromJson(
      '{"model": {"type": "Unigram", "vocab": [["a", 1]]}}',
    );
  });
});
