import { describe, it, expect } from 'vitest';
import { serializeTFJSWeights } from '../../../src/mmdnn/tfjs/serializer.js';

describe('TFJSSerializer', () => {
  it('should serialize weights', () => {
    const g: any = {
      initializers: ['w1'],
      tensors: {
        w1: {
          name: 'w1',
          shape: [1],
          dtype: 'float32',
          isInitializer: true,
          data: new Uint8Array(4),
        },
      },
    };
    const res = serializeTFJSWeights(g);
    expect(res.modelJson).toBeDefined();
    expect(res.weightsBin).toBeDefined();
  });
});
