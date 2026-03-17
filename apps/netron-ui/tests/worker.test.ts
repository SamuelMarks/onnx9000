import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as core from '@onnx9000/core';
import * as layout from '../src/layout/dag';
import { messageHandler } from '../src/parser/worker';

describe('Worker messageHandler', () => {
  let postMessageData: any = null;
  const postMessage = (d: any) => {
    postMessageData = d;
  };

  beforeEach(() => {
    postMessageData = null;
  });

  it('should parse buffer', async () => {
    vi.spyOn(core, 'parseModelProto').mockResolvedValue('Graph' as any);
    vi.spyOn(layout, 'computeLayout').mockReturnValue('Layout' as any);

    await messageHandler(
      { data: { type: 'PARSE_BUFFER', buffer: new Uint8Array([1]), direction: 'LR' } } as any,
      postMessage,
    );

    expect(postMessageData).toEqual({ type: 'PARSE_SUCCESS', graph: 'Graph', layout: 'Layout' });
  });

  it('should parse buffer without direction', async () => {
    vi.spyOn(core, 'parseModelProto').mockResolvedValue('Graph' as any);
    vi.spyOn(layout, 'computeLayout').mockReturnValue('Layout' as any);

    await messageHandler(
      { data: { type: 'PARSE_BUFFER', buffer: new Uint8Array([1]) } } as any,
      postMessage,
    );

    expect(postMessageData).toEqual({ type: 'PARSE_SUCCESS', graph: 'Graph', layout: 'Layout' });
  });

  it('should parse file', async () => {
    vi.spyOn(core, 'parseModelProto').mockResolvedValue('Graph' as any);
    vi.spyOn(layout, 'computeLayout').mockReturnValue('Layout' as any);

    await messageHandler({ data: { type: 'PARSE_FILE', file: new Blob() } } as any, postMessage);

    expect(postMessageData).toEqual({ type: 'PARSE_SUCCESS', graph: 'Graph', layout: 'Layout' });
  });

  it('should handle missing graph', async () => {
    vi.spyOn(core, 'parseModelProto').mockResolvedValue(null as any);

    await messageHandler({ data: { type: 'PARSE_FILE', file: new Blob() } } as any, postMessage);

    expect(postMessageData).toBeNull();
  });

  it('should emit error', async () => {
    vi.spyOn(core, 'parseModelProto').mockRejectedValue(new Error('Test error'));

    await messageHandler({ data: { type: 'PARSE_FILE', file: new Blob() } } as any, postMessage);

    expect(postMessageData).toEqual({ type: 'PARSE_ERROR', error: 'Test error' });
  });
});
