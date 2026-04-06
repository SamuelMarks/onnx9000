import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as core from '@onnx9000/core';
import * as layout from '../src/layout/dag';
import { messageHandler } from '../src/parser/worker';

describe('Worker messageHandler', () => {
  let postMessageData: Object = null;
  const postMessage = (d: Object) => {
    postMessageData = d;
  };

  beforeEach(() => {
    postMessageData = null;
  });

  it('should parse buffer', async () => {
    vi.spyOn(core, 'parseModelProto').mockResolvedValue('Graph' as Object);
    vi.spyOn(layout, 'computeLayout').mockReturnValue('Layout' as Object);

    await messageHandler(
      { data: { type: 'PARSE_BUFFER', buffer: new Uint8Array([1]), direction: 'LR' } } as Object,
      postMessage,
    );

    expect(postMessageData).toEqual({ type: 'PARSE_SUCCESS', graph: 'Graph', layout: 'Layout' });
  });

  it('should parse buffer without direction', async () => {
    vi.spyOn(core, 'parseModelProto').mockResolvedValue('Graph' as Object);
    vi.spyOn(layout, 'computeLayout').mockReturnValue('Layout' as Object);

    await messageHandler(
      { data: { type: 'PARSE_BUFFER', buffer: new Uint8Array([1]) } } as Object,
      postMessage,
    );

    expect(postMessageData).toEqual({ type: 'PARSE_SUCCESS', graph: 'Graph', layout: 'Layout' });
  });

  it('should parse file', async () => {
    vi.spyOn(core, 'parseModelProto').mockResolvedValue('Graph' as Object);
    vi.spyOn(layout, 'computeLayout').mockReturnValue('Layout' as Object);

    await messageHandler({ data: { type: 'PARSE_FILE', file: new Blob() } } as Object, postMessage);

    expect(postMessageData).toEqual({ type: 'PARSE_SUCCESS', graph: 'Graph', layout: 'Layout' });
  });

  it('should handle missing graph', async () => {
    vi.spyOn(core, 'parseModelProto').mockResolvedValue(null as Object);

    await messageHandler({ data: { type: 'PARSE_FILE', file: new Blob() } } as Object, postMessage);

    expect(postMessageData).toBeNull();
  });

  it('should emit error', async () => {
    vi.spyOn(core, 'parseModelProto').mockRejectedValue(new Error('Test error'));

    await messageHandler({ data: { type: 'PARSE_FILE', file: new Blob() } } as Object, postMessage);

    expect(postMessageData).toEqual({ type: 'PARSE_ERROR', error: 'Test error' });
  });
});
