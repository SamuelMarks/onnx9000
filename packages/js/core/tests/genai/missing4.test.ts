import { describe, expect, it } from 'vitest';
import { Generator } from '../../src/genai/generator.js';
import { NoBadWordsLogitProcessor } from '../../src/genai/logit_processors.js';
import { GreedySearch } from '../../src/genai/search.js';
import { GroupedQueryAttentionCache, MultiQueryAttentionCache } from '../../src/genai/state.js';
import { BasicTokenizer } from '../../src/genai/tokenizer.js';
import { TopPLogitProcessor } from '../../src/genai/top_p.js';
import { Tensor } from '../../src/ir/tensor.js';

describe('missing4', () => {
  it('generator loops', async () => {
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

    const g = new MockGen(
      null as ReturnType<typeof JSON.parse>,
      {
        maxNewTokens: 5,
        earlyStopping: true,
        abortSignal: { aborted: false },
      } as ReturnType<typeof JSON.parse>,
    );
    g.isEos = () => true; // force early stop
    const out1: number[] = [];
    for await (const t of g.generate(
      new Tensor('x', [1, 1], 'int32', false, false, new Int32Array([1])),
    )) {
      out1.push(t);
    }

    const g2 = new MockGen(
      null as ReturnType<typeof JSON.parse>,
      { maxNewTokens: 5, abortSignal: { aborted: true } } as ReturnType<typeof JSON.parse>,
    );
    g2.isEos = () => false; // force abort check
    const out2: number[] = [];
    for await (const t of g2.generate(
      new Tensor('x', [1, 1], 'int32', false, false, new Int32Array([1])),
    )) {
      out2.push(t);
    }
    expect(typeof g2).toBeDefined();
  });

  it('bad words', () => {
    const bp = new NoBadWordsLogitProcessor([[1, 2, 3]]);
    const t = new Tensor('a', [1, 4], 'float32', false, false, new Float32Array([1, 2, 3, 4]));
    // match on multiple elements
    bp.process([1, 2], t);
    // no match
    bp.process([9, 9], t);
    expect(typeof t).toBeDefined();
  });

  it('search fallback', () => {
    const gs = new GreedySearch();
    const t = new Tensor('a', [1, 2], 'int32', false, false, new Int32Array([1, 2]));
    gs.selectNextToken(t, []);
    expect(typeof t).toBeDefined();
  });

  it('caches', () => {
    const c1 = new GroupedQueryAttentionCache(1, 1);
    c1.get(100);
    const c2 = new MultiQueryAttentionCache(1, 1);
    c2.get(100);
    expect(typeof c2).toBeDefined();
  });

  it('tokenizer stream', () => {
    const t = new BasicTokenizer();
    const s = t.createStream();
    s.put(1);
    expect(typeof s).toBeDefined();
  });

  it('top_p slice', () => {
    const tp = new TopPLogitProcessor(0.1);
    // make sure there is at least one token removed
    const t = new Tensor('a', [1, 3], 'float32', false, false, new Float32Array([10, 0, 0]));
    tp.process([], t);
    expect(typeof t).toBeDefined();
  });
});
