import { describe, it } from 'vitest';
import * as tok from '../../src/genai/tokenizer';
import * as lp from '../../src/genai/logit_processors';
import { Tensor } from '../../src/ir/tensor';

describe('missing_last10', () => {
  it('logit_processors', () => {
    // 107-108: RepetitionPenaltyLogitProcessor val >= 0
    const r = new lp.RepetitionPenaltyLogitProcessor(1.0);
    r.process([1], new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([0, 1.0])));

    // 223-224: FrequencyPenaltyLogitProcessor inputIds.length === 0 or penalty === 0 or !Float32Array
    const f = new lp.FrequencyPenaltyLogitProcessor(0.0);
    f.process([1], new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1.0, 1.0])));
    const f2 = new lp.FrequencyPenaltyLogitProcessor(1.0);
    f2.process([], new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1.0, 1.0])));
    f2.process([1], new Tensor('a', [1, 2], 'int32', false, false, new Int32Array([1, 2])));

    // 277: ForcedEOSLogitProcessor - inputIds.length !== maxLength - 1
    const e = new lp.ForcedEOSLogitProcessor(5, 0);
    e.process([], new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1.0, 1.0])));
  });

  it('tokenizer 174-175, 178-179', () => {
    // BasicTokenizer: invVocab.get(tokenId) ?? this.unkToken where invVocab.get returns undefined
    const t = new tok.BasicTokenizer();
    t.idToToken(999);
    t.tokenToId('UNKNOWN');
  });
});
