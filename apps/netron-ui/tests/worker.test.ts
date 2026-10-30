import * as core from '@onnx9000/core';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import * as layout from '../src/layout/dag.js';
import { messageHandler } from '../src/parser/worker.js';

type WorkerEvent = Parameters<typeof messageHandler>[0];

describe('Worker messageHandler', () => {
  let postMessageData: unknown = null;
  const postMessage = (d: unknown) => {
    postMessageData = d;
  };

  beforeEach(() => {
    postMessageData = null;
  });

  it('should parse buffer', async () => {
    vi.spyOn(core, 'parseModelProto').mockResolvedValue(
      'Graph' as Awaited<ReturnType<typeof core.parseModelProto>>,
    );
    vi.spyOn(layout, 'computeLayout').mockReturnValue(
      'Layout' as ReturnType<typeof layout.computeLayout>,
    );

    await messageHandler(
      {
        data: {
          type: 'PARSE_BUFFER',
          buffer: new Uint8Array([1]),
          direction: 'LR',
        },
      } as WorkerEvent,
      postMessage,
    );

    expect(postMessageData).toEqual({
      type: 'PARSE_SUCCESS',
      graph: 'Graph',
      layout: 'Layout',
    });
  });

  it('should parse buffer without direction', async () => {
    vi.spyOn(core, 'parseModelProto').mockResolvedValue(
      'Graph' as Awaited<ReturnType<typeof core.parseModelProto>>,
    );
    vi.spyOn(layout, 'computeLayout').mockReturnValue(
      'Layout' as ReturnType<typeof layout.computeLayout>,
    );

    await messageHandler(
      { data: { type: 'PARSE_BUFFER', buffer: new Uint8Array([1]) } } as WorkerEvent,
      postMessage,
    );

    expect(postMessageData).toEqual({
      type: 'PARSE_SUCCESS',
      graph: 'Graph',
      layout: 'Layout',
    });
  });

  it('should parse file', async () => {
    vi.spyOn(core, 'parseModelProto').mockResolvedValue(
      'Graph' as Awaited<ReturnType<typeof core.parseModelProto>>,
    );
    vi.spyOn(layout, 'computeLayout').mockReturnValue(
      'Layout' as ReturnType<typeof layout.computeLayout>,
    );

    await messageHandler(
      { data: { type: 'PARSE_FILE', file: new Blob() } } as WorkerEvent,
      postMessage,
    );

    expect(postMessageData).toEqual({
      type: 'PARSE_SUCCESS',
      graph: 'Graph',
      layout: 'Layout',
    });
  });

  it('should handle missing graph', async () => {
    vi.spyOn(core, 'parseModelProto').mockResolvedValue(
      null as Awaited<ReturnType<typeof core.parseModelProto>>,
    );

    await messageHandler(
      { data: { type: 'PARSE_FILE', file: new Blob() } } as WorkerEvent,
      postMessage,
    );

    expect(postMessageData).toBeNull();
  });

  it('should emit error', async () => {
    vi.spyOn(core, 'parseModelProto').mockRejectedValue(new Error('Test error'));

    await messageHandler(
      { data: { type: 'PARSE_FILE', file: new Blob() } } as WorkerEvent,
      postMessage,
    );

    expect(postMessageData).toEqual({
      type: 'PARSE_ERROR',
      error: 'Test error',
    });
  });
});
