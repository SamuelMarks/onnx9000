import { describe, expect, it } from 'vitest';
import * as index from '../src/index.js';

describe('index', () => {
  it('should export modules', () => {
    expect(index.TFLiteExporter).toBeDefined();
  });
});
