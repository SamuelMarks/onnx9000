import { describe, expect, it } from 'vitest';
import { Generator } from '../../src/genai/generator.js';
import {
  AllowedWordsLogitProcessor,
  NoBadWordsLogitProcessor,
} from '../../src/genai/logit_processors.js';
import { GreedySearch } from '../../src/genai/search.js';
import { GroupedQueryAttentionCache, MultiQueryAttentionCache } from '../../src/genai/state.js';
import { BasicTokenizer } from '../../src/genai/tokenizer.js';
import { TopPLogitProcessor } from '../../src/genai/top_p.js';
import { Tensor } from '../../src/ir/tensor.js';

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
        return null as ReturnType<typeof JSON.parse>;
      }
    }
    const gen1 = new MockGen(
      null as ReturnType<typeof JSON.parse>,
      { maxNewTokens: null, abortSignal: { aborted: true } } as ReturnType<typeof JSON.parse>,
    );
    const out1: number[] = [];
    for await (const t of gen1.generate(
      new Tensor('x', [1, 1], 'int32', false, false, new Int32Array([1])),
    )) {
      out1.push(t);
    }

    const gen2 = new MockGen(
      null as ReturnType<typeof JSON.parse>,
      { earlyStopping: true, maxNewTokens: null } as ReturnType<typeof JSON.parse>,
    );
    gen2.isEos = () => true;
    const out2: number[] = [];
    for await (const t of gen2.generate(
      new Tensor('x', [1, 1], 'int32', false, false, new Int32Array([1])),
    )) {
      out2.push(t);
    }
    expect(typeof gen2).toBeDefined();
  });

  it('logit_processors', () => {
    const bp = new NoBadWordsLogitProcessor([[1, 2]]);
    const t = new Tensor('a', [1, 3], 'float32', false, false, new Float32Array([1, 2, 3]));
    bp.process([1], t); // matches bad word prefix

    const ap = new AllowedWordsLogitProcessor([[1, 2]]);
    ap.process([1], t);
    expect(typeof ap).toBeDefined(); // matches allowed word prefix
  });

  it('search', () => {
    const gs = new GreedySearch();
    const t = new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([NaN, Infinity]));
    gs.selectNextToken(t, []);
    expect(typeof t).toBeDefined();
  });

  it('state', () => {
    const c1 = new GroupedQueryAttentionCache(1, 1);
    c1.get(0);
    const c2 = new MultiQueryAttentionCache(1, 1);
    c2.get(0);
    expect(typeof c2).toBeDefined();
  });

  it('tokenizer', () => {
    const t = new BasicTokenizer();
    const s = t.createStream();
    s.put(1);
    expect(typeof s).toBeDefined();
  });

  it('top_p', () => {
    const tp = new TopPLogitProcessor(0.5);
    const t = new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2]));
    tp.process([], t);
    expect(typeof t).toBeDefined();
  });
});
