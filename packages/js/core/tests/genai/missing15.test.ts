import { describe, it } from 'vitest';
import * as search from '../../src/genai/search';
import * as tok from '../../src/genai/tokenizer';
import * as lp from '../../src/genai/logit_processors';
import * as state from '../../src/genai/state';
import { Tensor } from '../../src/ir/tensor';

describe('missing15', () => {
  it('search 23', () => {
    const gs = new search.GreedySearch();
    const t = new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, NaN]));
    gs.selectNextToken(t, []);

    const t2 = new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, Infinity]));
    gs.selectNextToken(t2, []);
  });

  it('tokenizer', () => {
    const w = new tok.WordPieceTokenizer(new Map([['a', 1]]));
    w.idToToken(9999);
    w.tokenToId('UNKNOWN');

    const u = new tok.UnigramTokenizer(new Map([['a', 1]]));
    u.encodeBatch(['a']);
    u.decodeBatch([[1]]);
  });
});
