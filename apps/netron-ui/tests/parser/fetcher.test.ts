import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fetchAndParseModel } from '../../src/parser/fetcher.ts';

// Mock the named export
vi.mock('@onnx9000/core', async () => {
  return {
    BlobReader: class { constructor(b: any) {} },
    parseModelProto: vi.fn().mockResolvedValue({ name: 'mocked_graph' }),
    Graph: class { constructor() {} }
  };
});

describe('fetchAndParseModel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should fetch and parse a model successfully', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ 'content-length': '9' }),
      body: {
        getReader: () => {
          let done = false;
          return {
            read: () => {
              if (done) return Promise.resolve({ done: true, value: undefined });
              done = true;
              return Promise.resolve({ done: false, value: new Uint8Array([1, 2, 3]) });
            }
          };
        }
      }
    });

    const progressCb = vi.fn();
    const result = await fetchAndParseModel('http://example.com/model.onnx', progressCb);
    expect(result.name).toBe('mocked_graph');
    expect(progressCb).toHaveBeenCalled();
  });

  it('should transform github blob URLs', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers(),
      body: {
        getReader: () => ({ read: () => Promise.resolve({ done: true, value: undefined }) })
      }
    });
    
    await fetchAndParseModel('https://github.com/user/repo/blob/main/model.onnx');
    expect(global.fetch).toHaveBeenCalledWith('https://raw.githubusercontent.com/user/repo/main/model.onnx');
  });
});
