import { describe, expect, it } from 'vitest';
import {
  ModelParams,
  GeneratorParams,
  State,
  KVCache,
  ContinuousKVCache,
  PagedKVCache,
  MultiHeadAttentionCache,
  GroupedQueryAttentionCache,
  MultiQueryAttentionCache,
  SequenceBatchingKVCache,
  CrossAttentionCache,
  SlidingWindowKVCache,
  PositionalEmbeddingUtils,
  Generator,
  Model,
  SequenceTensorUtils,
} from '../../src/genai/index.js';
import { Tensor, Graph } from '../../src/index.js';

class MockKVCache implements KVCache {
  private cache: Map<number, { keys: Tensor; values: Tensor }> = new Map();

  clear() {
    this.cache.clear();
  }
  update(keys: Tensor, values: Tensor, layerIdx: number) {
    this.cache.set(layerIdx, { keys, values });
  }
  get(layerIdx: number) {
    return this.cache.get(layerIdx) || null;
  }
}

class MockGenerator extends Generator {
  async computeLogits(inputIds: Tensor): Promise<Tensor> {
    return new Tensor('test', [1, 10], 1, false, false, new Float32Array(10));
  }
  computeLogitsSync(inputIds: Tensor): Tensor {
    return new Tensor('test', [1, 10], 1, false, false, new Float32Array(10));
  }
  async prefill(promptIds: Tensor): Promise<Tensor> {
    return new Tensor('test', [1, 10], 1, false, false, new Float32Array(10));
  }
  async decodeStep(tokenId: number): Promise<Tensor> {
    return new Tensor('test', [1, 10], 1, false, false, new Float32Array(10));
  }
}

class MockModel extends Model {
  createGenerator(params: GeneratorParams): Generator {
    const state = new State({} as Graph, new MockKVCache());
    return new MockGenerator(state, params);
  }
}

describe('GenAI Core', () => {
  it('ContinuousKVCache', () => {
    const cache = new ContinuousKVCache();
    const keys = new Tensor('test', [1, 2, 64], 1, false, false, new Float32Array(128));
    const values = new Tensor('test', [1, 2, 64], 1, false, false, new Float32Array(128));

    cache.update(keys, values, 0);
    expect(cache.get(0)).toBeTruthy();

    cache.clear();
    expect(cache.get(0)).toBeNull();
  });

  it('PagedKVCache', () => {
    const cache = new PagedKVCache(16);
    const keys = new Tensor('test', [1, 2, 64], 1, false, false, new Float32Array(128));
    const values = new Tensor('test', [1, 2, 64], 1, false, false, new Float32Array(128));

    cache.update(keys, values, 0);
    expect(cache.get(0)).toBeTruthy();

    cache.clear();
    expect(cache.get(0)).toBeNull();
  });

  it('MultiHeadAttentionCache', () => {
    const cache = new MultiHeadAttentionCache(12, 64);
    const keys = new Tensor('test', [1, 12, 2, 64], 1, false, false, new Float32Array(12 * 2 * 64));
    const values = new Tensor(
      'test',
      [1, 12, 2, 64],
      1,
      false,
      false,
      new Float32Array(12 * 2 * 64),
    );
    cache.update(keys, values, 0);
    expect(cache.get(0)).toBeTruthy();
  });

  it('GroupedQueryAttentionCache', () => {
    const cache = new GroupedQueryAttentionCache(4, 64);
    const keys = new Tensor('test', [1, 4, 2, 64], 1, false, false, new Float32Array(4 * 2 * 64));
    const values = new Tensor('test', [1, 4, 2, 64], 1, false, false, new Float32Array(4 * 2 * 64));
    cache.update(keys, values, 0);
    expect(cache.get(0)).toBeTruthy();
  });

  it('MultiQueryAttentionCache', () => {
    const cache = new MultiQueryAttentionCache(64);
    const keys = new Tensor('test', [1, 1, 2, 64], 1, false, false, new Float32Array(1 * 2 * 64));
    const values = new Tensor('test', [1, 1, 2, 64], 1, false, false, new Float32Array(1 * 2 * 64));
    cache.update(keys, values, 0);
    expect(cache.get(0)).toBeTruthy();
  });

  it('SequenceBatchingKVCache', () => {
    const cache = new SequenceBatchingKVCache();
    const keys = new Tensor(
      'test',
      [1, 1, 2, 64],
      1,
      false,
      false,
      new Float32Array(1 * 1 * 2 * 64),
    );
    const values = new Tensor(
      'test',
      [1, 1, 2, 64],
      1,
      false,
      false,
      new Float32Array(1 * 1 * 2 * 64),
    );
    cache.update(keys, values, 0);
    expect(cache.get(0)).toBeTruthy();
    cache.clear();
    expect(cache.get(0)).toBeNull();
  });

  it('CrossAttentionCache', () => {
    const cache = new CrossAttentionCache();
    const keys = new Tensor(
      'test',
      [1, 1, 2, 64],
      1,
      false,
      false,
      new Float32Array(1 * 1 * 2 * 64),
    );
    const values = new Tensor(
      'test',
      [1, 1, 2, 64],
      1,
      false,
      false,
      new Float32Array(1 * 1 * 2 * 64),
    );
    cache.update(keys, values, 0);
    expect(cache.get(0)).toBeTruthy();
    cache.clear();
    expect(cache.get(0)).toBeNull();
  });

  it('SlidingWindowKVCache', () => {
    const cache = new SlidingWindowKVCache(2048);
    const keys = new Tensor(
      'test',
      [1, 1, 2, 64],
      1,
      false,
      false,
      new Float32Array(1 * 1 * 2 * 64),
    );
    const values = new Tensor(
      'test',
      [1, 1, 2, 64],
      1,
      false,
      false,
      new Float32Array(1 * 1 * 2 * 64),
    );
    cache.update(keys, values, 0);
    expect(cache.get(0)).toBeTruthy();
    cache.clear();
    expect(cache.get(0)).toBeNull();
  });

  it('PositionalEmbeddingUtils', () => {
    const q = new Tensor('q', [1, 1, 2, 64], 1, false, false, new Float32Array(1 * 1 * 2 * 64));
    const k = new Tensor('k', [1, 1, 2, 64], 1, false, false, new Float32Array(1 * 1 * 2 * 64));

    const [qRope, kRope] = PositionalEmbeddingUtils.applyRoPE(q, k, 2);
    expect(qRope.shape).toEqual(q.shape);
    expect(kRope.shape).toEqual(k.shape);

    const scores = new Tensor(
      'scores',
      [1, 2, 2, 2],
      1,
      false,
      false,
      new Float32Array(1 * 2 * 2 * 2),
    );
    const scoresAlibi = PositionalEmbeddingUtils.applyALiBi(scores, 2);
    expect(scoresAlibi.shape).toEqual(scores.shape);
  });

  it('types and state setup', () => {
    const cache = new MockKVCache();
    const state = new State({} as Graph, cache);

    expect(state.currentLength).toBe(0);
    expect(state.isPrefill).toBe(true);

    const keys = new Tensor('test', [1, 2, 64], 1, false, false, new Float32Array(128));
    const values = new Tensor('test', [1, 2, 64], 1, false, false, new Float32Array(128));

    cache.update(keys, values, 0);
    expect(cache.get(0)).toBeTruthy();

    state.reset();
    expect(cache.get(0)).toBeNull();
  });

  it('SequenceTensorUtils.expandSequenceDimension', () => {
    const data = new Float32Array(2 * 3 * 4);
    data.fill(1.0);
    const tensor = new Tensor('test', [2, 3, 4], 1, false, false, data);

    const expanded = SequenceTensorUtils.expandSequenceDimension(tensor, 5);
    expect(expanded.shape).toEqual([2, 5, 4]);
    expect(expanded.data.length).toBe(2 * 5 * 4);
  });

  it('Generator loop', async () => {
    const state = new State({} as Graph, new MockKVCache());
    const params: GeneratorParams = {
      maxLength: 10,
      maxNewTokens: 3,
      earlyStopping: false,
      numBeams: 1,
      temperature: 1.0,
      topK: 50,
      topP: 1.0,
      repetitionPenalty: 1.0,
      numReturnSequences: 1,
      doSample: false,
    };
    const gen = new MockGenerator(state, params);
    const prompt = new Tensor('test', [1, 2], 6, false, false, new Int32Array(2));

    const tokens = [];
    for await (const token of gen.generate(prompt)) {
      tokens.push(token);
    }
    expect(tokens.length).toBe(3);
  });

  it('Model generate high level API', async () => {
    const mParams: ModelParams = {
      maxSequenceLength: 1024,
      numHiddenLayers: 1,
      numAttentionHeads: 1,
      numKeyValueHeads: 1,
      hiddenSize: 64,
      vocabSize: 10,
      eosTokenId: 9,
    };
    const params: GeneratorParams = {
      maxLength: 10,
      maxNewTokens: 4,
      earlyStopping: false,
      numBeams: 1,
      temperature: 1.0,
      topK: 50,
      topP: 1.0,
      repetitionPenalty: 1.0,
      numReturnSequences: 1,
      doSample: false,
    };

    const model = new MockModel(mParams);
    const prompt = new Tensor('test', [1, 2], 6, false, false, new Int32Array(2));

    const tokens = [];
    for await (const token of model.generate(prompt, params)) {
      tokens.push(token);
    }
    expect(tokens.length).toBe(4);
  });

  it('Zero-length prompt', async () => {
    const state = new State({} as Graph, new MockKVCache());
    const params: GeneratorParams = {
      maxLength: 10,
      maxNewTokens: 3,
      earlyStopping: false,
      numBeams: 1,
      temperature: 1.0,
      topK: 50,
      topP: 1.0,
      repetitionPenalty: 1.0,
      numReturnSequences: 1,
      doSample: false,
    };
    const gen = new MockGenerator(state, params);
    const prompt = new Tensor('test', [1, 0], 6, false, false, new Int32Array(0));

    const tokens = [];
    for await (const token of gen.generate(prompt)) {
      tokens.push(token);
    }
    expect(tokens.length).toBe(3);
  });
});
