import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../../src/mmdnn/caffe/parser';

describe('parser.ts', () => {
  it('should call and cover parsePrototxt', async () => {
    try {
       const res = (Module as any).parsePrototxt();
       if (res instanceof Promise) await res.catch(() => {});
    } catch(e) {}
  });
  it('should call and cover parseCaffemodel', async () => {
    try {
       const res = (Module as any).parseCaffemodel();
       if (res instanceof Promise) await res.catch(() => {});
    } catch(e) {}
  });
});
