import { describe, it, expect } from 'vitest';
import { parseH2O } from '../../../src/mmdnn/h2o/parser.js';

describe('H2O Parser', () => {
  it('should parse valid JSON MOJO mock', () => {
    const jsonStr = JSON.stringify({ algo: 'xgboost' });
    const parsed = parseH2O(jsonStr);
    expect(parsed.algo).toBe('xgboost');
  });

  it('should handle invalid string gracefully', () => {
    const parsed = parseH2O('invalid');
    expect(parsed).toEqual({});
  });
});
