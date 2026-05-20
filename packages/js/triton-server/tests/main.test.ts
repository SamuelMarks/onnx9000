import { describe, it, expect } from 'vitest';
import { TritonServer } from '../src/index';

describe('TritonServer', () => {
  it('should process correctly', () => {
    const obj = new TritonServer();
    expect(obj.process('test')).toBe('Triton Server processed test');
  });
});
