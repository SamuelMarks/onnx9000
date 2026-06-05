import { describe, it, expect, vi } from 'vitest';
import { parseKerasH5 } from '../../src/keras/h5-parser.js';

vi.mock('jsfive', () => ({
  File: class {
    constructor() {}
    attrs = {
      model_config: '{"config": {}}',
      keras_version: '2',
      backend: 'tf',
    };
  },
}));

describe('h5-parser', () => {
  it('should parse h5', () => {
    const res = parseKerasH5(new ArrayBuffer(10));
    expect(res.modelConfig).toBeDefined();
    expect(res.kerasVersion).toBe('2');
  });
});
