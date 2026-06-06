import { describe, it, expect, vi } from 'vitest';
import { messageHandler } from '../../src/parser/worker.ts';

vi.mock('@onnx9000/core', async () => {
  return {
    BlobReader: class { constructor(b: any) {} },
    BufferReader: class { constructor(b: any) {} },
    parseModelProto: vi.fn().mockResolvedValue({ name: 'worker_graph' }),
  };
});

vi.mock('../../src/layout/dag.ts', async () => {
  return {
    computeLayout: vi.fn().mockReturnValue({ width: 100, height: 100 }),
  };
});

describe('worker messageHandler', () => {
  it('should handle PARSE_FILE', async () => {
    const postMessage = vi.fn();
    const event = {
      data: {
        type: 'PARSE_FILE',
        file: new Blob(['mock']),
        direction: 'TB'
      }
    } as any;
    
    await messageHandler(event, postMessage);
    
    expect(postMessage).toHaveBeenCalledWith({
      type: 'PARSE_SUCCESS',
      graph: { name: 'worker_graph' },
      layout: { width: 100, height: 100 }
    });
  });

  it('should handle parsing errors', async () => {
    const postMessage = vi.fn();
    const event = {
      data: {
        type: 'PARSE_BUFFER',
        buffer: null,
      }
    } as any;
    
    await messageHandler(event, postMessage);
    
    expect(postMessage).toHaveBeenCalledWith(expect.objectContaining({
      type: 'PARSE_ERROR'
    }));
  });
});
