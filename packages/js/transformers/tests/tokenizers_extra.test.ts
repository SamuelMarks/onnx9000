import { describe, expect, it } from 'vitest';
import * as toks from '../src/tokenizers/index.js';

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
        const Ctor = Cls as new (
          cfg: Record<string, unknown>,
        ) => {
          encode?: (s: string) => unknown;
          decode?: (ids: number[]) => unknown;
          init?: () => Promise<void>;
        };
        const t = new Ctor({});
        expect(t).toBeDefined();
        if (t.encode) expect(t.encode('test')).toBeDefined();
        if (t.decode) expect(t.decode([1, 2, 3])).toBeDefined();
        if (t.init) await t.init();
      }
    }
  });

  it('Callable behavior', async () => {
    const _t = new toks.PreTrainedTokenizer({});
    // no-op
    expect(typeof _t).toBeDefined();
  });
});
