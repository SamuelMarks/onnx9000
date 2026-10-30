import { describe, expect, it } from 'vitest';
import { WorkerManager } from '../../src/core/WorkerManager';

describe('WorkerManager.ts', () => {
  it('should instantiate and cover WorkerManager', () => {
    const obj = WorkerManager.getInstance();
    expect(obj).toBeDefined();
  });
});
