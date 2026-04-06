import { describe, it, expect } from 'vitest';
import { Generator } from '../../src/genai/generator';
import { Tensor } from '../../src/ir/tensor';
import {
  NoBadWordsLogitProcessor,
  AllowedWordsLogitProcessor,
} from '../../src/genai/logit_processors';
import { GreedySearch } from '../../src/genai/search';
import { GroupedQueryAttentionCache, MultiQueryAttentionCache } from '../../src/genai/state';
import { BasicTokenizer } from '../../src/genai/tokenizer';
import { TopPLogitProcessor } from '../../src/genai/top_p';

describe('missing3', () => {
  it('generator', async () => {
    class MockGen extends Generator {
      async prefill() {
        return new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2]));
      }
      async decodeStep() {
        return new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2]));
      }
      sample() {
        return 1;
      }
      createModel() {
        return null as Object;
      }
    }
    const gen1 = new MockGen(
      null as Object,
      { maxNewTokens: null, abortSignal: { aborted: true } } as Object,
    );
    for await (const t of gen1.generate(
      new Tensor('x', [1, 1], 'int32', false, false, new Int32Array([1])),
    )) {
    }

    const gen2 = new MockGen(null as Object, { earlyStopping: true, maxNewTokens: null } as Object);
    gen2.isEos = () => true;
    for await (const t of gen2.generate(
      new Tensor('x', [1, 1], 'int32', false, false, new Int32Array([1])),
    )) {
    }
  });

  it('logit_processors', () => {
    const bp = new NoBadWordsLogitProcessor([[1, 2]]);
    const t = new Tensor('a', [1, 3], 'float32', false, false, new Float32Array([1, 2, 3]));
    bp.process([1], t); // matches bad word prefix

    const ap = new AllowedWordsLogitProcessor([[1, 2]]);
    ap.process([1], t); // matches allowed word prefix
  });

  it('search', () => {
    const gs = new GreedySearch();
    const t = new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([NaN, Infinity]));
    gs.selectNextToken(t, []);
  });

  it('state', () => {
    const c1 = new GroupedQueryAttentionCache(1, 1);
    c1.get(0);
    const c2 = new MultiQueryAttentionCache(1, 1);
    c2.get(0);
  });

  it('tokenizer', () => {
    const t = new BasicTokenizer();
    const s = t.createStream();
    s.put(1);
  });

  it('top_p', () => {
    const tp = new TopPLogitProcessor(0.5);
    const t = new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2]));
    tp.process([], t);
  });
});
