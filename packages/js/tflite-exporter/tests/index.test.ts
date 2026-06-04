import * as index from '../src/index.js';
import { describe, it, expect } from 'vitest';

describe('index', () => {
  it('should export modules', () => {
    expect(index.TFLiteExporter).toBeDefined();
  });
});
