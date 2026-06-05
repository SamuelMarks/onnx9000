import { describe, it, expect } from 'vitest';
import { parseH2O } from '../../../src/mmdnn/h2o/parser.js';

describe('h2o parser', () => {
  it('should parse json', () => {
    const res = parseH2O('{"algo": "xgboost"}');
    expect(res.algo).toBe('xgboost');
  });
});
