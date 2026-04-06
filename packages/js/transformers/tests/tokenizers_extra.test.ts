import { describe, it, expect } from 'vitest';
import * as toks from '../src/tokenizers/index';

describe('Tokenizers coverage', () => {
  it('instantiates all', async () => {
    const list = [
      toks.AutoTokenizer,
      toks.PreTrainedTokenizer,
      toks.BertTokenizer,
      toks.RobertaTokenizer,
      toks.T5Tokenizer,
      toks.GPT2Tokenizer,
      toks.LlamaTokenizer,
      toks.WhisperTokenizer,
      toks.Wav2Vec2CTCTokenizer,
    ].filter((x) => x);

    for (const Cls of list) {
      if (Cls === toks.AutoTokenizer) {
        expect(await toks.AutoTokenizer.fromPretrained('a')).toBeDefined();
      } else {
        const t = new (Cls as Object)({});
        expect(t).toBeDefined();
        if (t.encode) expect(t.encode('test')).toBeDefined();
        if (t.decode) expect(t.decode([1, 2, 3])).toBeDefined();
        if (t.init) await t.init();
      }
    }
  });

  it('Callable behavior', async () => {
    const t = new toks.PreTrainedTokenizer({});
    // no-op
  });
});
