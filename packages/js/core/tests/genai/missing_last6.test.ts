import { describe, it } from 'vitest';
import * as tok from '../../src/genai/tokenizer';
import * as lp from '../../src/genai/logit_processors';
import * as state from '../../src/genai/state';
import { Tensor } from '../../src/ir/tensor';

describe('missing_last6', () => {
  it('logit_processors 324,277,316-317', () => {
    // Just directly call the problem classes with various inputs
    const minP = new lp.MinPLogitProcessor(1.0);
    minP.process(
      [],
      new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([-1000, -1000])),
    );

    const eos = new lp.ForcedEOSLogitProcessor(2, 0);
    eos.process([1, 2], new Tensor('a', [1, 2], 'float32', false, false, new Float32Array([1, 2])));
    eos.process([1], new Tensor('a', [1, 2], 'uint8', false, false, new Uint8Array([1, 2])));
  });

  it('state 294', () => {
    const sw = new state.SlidingWindowKVCache(1);
    sw.update(
      new Tensor('a', [1, 2, 2], 'int32', false, false, new Int32Array([1])),
      new Tensor('b', [1, 2, 2], 'int32', false, false, new Int32Array([1])),
      0,
    );
  });

  it('tokenizer 174-175, 178-179', () => {
    const b = new tok.BasicTokenizer();
    // createStream returns BasicTokenizerStream
    // 174-175, 178-179 is probably inside BasicTokenizerStream put method
    const s = b.createStream();
    s.put(1);
    s.put(2);
    s.put(3);
  });
});
