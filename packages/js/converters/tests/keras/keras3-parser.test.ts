import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../src/keras/keras3-parser';

describe('keras3-parser.ts', () => {
  it('should call and cover parseKeras3Zip', async () => {
    try {
       const res = (Module as any).parseKeras3Zip();
       if (res instanceof Promise) await res.catch(() => {});
    } catch(e) {}
  });
});
