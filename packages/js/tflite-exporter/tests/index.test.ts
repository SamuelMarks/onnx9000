import * as index from '../src/index';
import { describe, it, expect } from 'vitest';

describe('index', () => {
  it('should export modules', () => {
    expect(index.TFLiteExporter).toBeDefined();
  });
});
