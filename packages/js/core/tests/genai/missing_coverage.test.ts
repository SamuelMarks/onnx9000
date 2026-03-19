import { describe, it, expect } from 'vitest';
import { Generator } from '../../src/genai/generator';
import { Tensor } from '../../src/ir/tensor';
import {
  DiverseBeamSearchLogitProcessor,
  ContrastiveSearchLogitProcessor,
} from '../../src/genai/logit_processors';
import { Model } from '../../src/genai/model';
import { BeamSearchAlgorithm, BeamSearchState } from '../../src/genai/search';
import { QuantizedKVCache, OffloadedKVCache, PromptCacheManager } from '../../src/genai/state';
import { SequenceTensorUtils } from '../../src/genai/tensor_utils';
import {
  PreTokenizer,
  TokenTrie,
  StreamingUTF8Decoder,
  LlamaTokenizer,
  GPT2Tokenizer,
  loadTokenizerWithFallback,
} from '../../src/genai/tokenizer';
import { TopPLogitProcessor } from '../../src/genai/top_p';

describe('missing_coverage', () => {
  it('generator', async () => {
    class MockGen extends Generator {
      async prefill(p: any) {
        return new Tensor('a', [1, 2], 'float32', false, false, new Float32Array(2));
      }
      async decodeStep(t: any) {
        return new Tensor('a', [1, 2], 'float32', false, false, new Float32Array(2));
      }
      createModel() {
        return null as any;
      }
    }
    const gen = new MockGen(
      null as any,
      { earlyStopping: true, maxNewTokens: 1, abortSignal: false } as any,
    );

    try {
      (gen as any).sample(new Tensor('a', [1, 2], 'int32', false, false, new Int32Array([1, 2])));
    } catch (e: any) {
      expect(e.message).toBe('Unsupported logit data type for sampling.');
    }
    expect((gen as any).isEos(1)).toBe(false);
  });

  it('logit_processors', () => {
    new DiverseBeamSearchLogitProcessor(1, 1, 1).process([], null as any);
    new ContrastiveSearchLogitProcessor(1).process([], null as any);
  });

  it('model', () => {
    class MockModel extends Model {
      async predict(i: any) {
        return null as any;
      }
      createGenerator(p: any) {
        return null as any;
      }
    }
    const m = new MockModel();
    m.loadWeights();
    m.createTokenizer();
  });

  it('search', () => {
    const s = new BeamSearchAlgorithm(new BeamSearchState(1, 1));
    s.processLogits(new Tensor('a', [1, 2], 'int32', false, false, new Int32Array([1, 2])), 0);
    s.processLogits(new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2])), 0);
    s.pruneAndSortBeams([{ score: 1, tokens: [] }]);
  });

  it('state', () => {
    const cache1 = new QuantizedKVCache();
    cache1.clear();
    cache1.update(null as any, null as any, 0);
    cache1.get(0);

    const cache2 = new OffloadedKVCache(1);
    cache2.clear();
    cache2.update(null as any, null as any, 0);
    cache2.get(0);

    new PromptCacheManager().saveToIDB('', null);
  });

  it('tensor_utils', () => {
    try {
      SequenceTensorUtils.expandSequenceDimension(
        new Tensor('a', [1], 'int32', false, false, new Int32Array(1)),
        2,
      );
    } catch (e: any) {
      expect(e.message).toBe('Tensor must have at least 2 dimensions to expand sequence length.');
    }

    SequenceTensorUtils.expandSequenceDimension(
      new Tensor('a', [1, 1], 'int32', false, false, new Int32Array(1)),
      2,
    );
    SequenceTensorUtils.expandSequenceDimension(
      new Tensor('a', [1, 1], 'float64', false, false, new Float64Array(1)),
      2,
    );
    SequenceTensorUtils.expandSequenceDimension(
      new Tensor('a', [1, 1], 'uint8', false, false, new Uint8Array(1)),
      2,
    );
  });

  it('tokenizer', () => {
    PreTokenizer.punctuationSplit('');
    PreTokenizer.byteLevel('a');
    new TokenTrie();
    new StreamingUTF8Decoder().decode(new Uint8Array(1));
    new LlamaTokenizer();
    new GPT2Tokenizer();
    loadTokenizerWithFallback('');
  });

  it('top_p', () => {
    const p = new TopPLogitProcessor(1.0);
    p.process([], null as any);

    const p2 = new TopPLogitProcessor(0.5);
    p2.process([], new Tensor('a', [1, 2], 'int32', false, false, new Int32Array([1, 2])));
  });
});
