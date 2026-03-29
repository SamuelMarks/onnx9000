import { describe, it, expect, vi } from 'vitest';
import { createServer, DynamicBatcher, MemoryManager, globalMetrics } from '../src/index';

describe('ONNX9000 Serve API', () => {
  const server = createServer();

  it('201. Unit Test: Boot server, process KServe JSON request', async () => {
    const kservePayload = {
      inputs: [
        {
          name: 'input_0',
          datatype: 'FP32',
          shape: [1, 3, 224, 224],
          data: new Array(1 * 3 * 224 * 224).fill(0.1),
        },
      ],
    };

    const req = new Request('http://localhost/v2/models/resnet/infer', {
      method: 'POST',
      body: JSON.stringify(kservePayload),
      headers: { 'Content-Type': 'application/json' },
    });

    const res = await server.fetch(req);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.model_name).toBe('resnet');
    expect(body.outputs.length).toBe(1);
    expect(body.outputs[0].name).toBe('input_0'); // mock echo behavior
  });

  it('202. Unit Test: Boot server, process OpenAI Chat Completion stream', async () => {
    const openaiPayload = {
      model: 'tinyllama',
      messages: [{ role: 'user', content: 'Hello' }],
      stream: true,
    };

    const req = new Request('http://localhost/v1/chat/completions', {
      method: 'POST',
      body: JSON.stringify(openaiPayload),
      headers: { 'Content-Type': 'application/json' },
    });

    const res = await server.fetch(req);
    expect(res.status).toBe(200);
    expect(res.headers.get('content-type')).toBe('text/event-stream');

    // Quick stream parse
    const reader = res.body?.getReader();
    expect(reader).toBeDefined();
    const { value } = await reader!.read();
    const text = new TextDecoder().decode(value);
    expect(text).toContain('data: {"id":"chatcmpl-123"');
  });

  it('203. Unit Test: Execute 5 simultaneous requests natively and ensure batching triggers', async () => {
    let batchExecutions = 0;
    const batcher = new DynamicBatcher(
      async (batch) => {
        batchExecutions++;
        return batch.map((b) => b.payload);
      },
      { maxBatchSize: 5, batchTimeoutMs: 50 },
    );

    const promises = [];
    for (let i = 0; i < 5; i++) {
      promises.push(batcher.add({ val: i }));
    }

    const results = await Promise.all(promises);
    expect(results.length).toBe(5);
    expect(batchExecutions).toBe(1); // Exact 1 batch triggered
  });

  it('204. Validate JSON parsing strictness', async () => {
    const req = new Request('http://localhost/v2/models/bad/infer', {
      method: 'POST',
      body: '{"inputs": [{ "name": "a", "shape": [1], "datatype": "INVALID_TYPE" }]}',
      headers: { 'Content-Type': 'application/json' },
    });
    const res = await server.fetch(req);
    expect(res.status).toBe(400); // Bad Request from KServe validation
  });

  it('207. Validate the Prometheus metrics formatting', async () => {
    globalMetrics.incrementRequests();
    const req = new Request('http://localhost/metrics');
    const res = await server.fetch(req);
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toContain('onnx9000_inference_request_total');
  });

  it('208. Test memory eviction natively', async () => {
    const manager = new MemoryManager({ maxRamBytes: 1000, maxRamPercent: 1.0 });
    const instanceA = {
      id: 'A',
      sizeBytes: 600,
      lastUsed: 0,
      buffer: new ArrayBuffer(0),
      unload: vi.fn(),
    };
    const instanceB = {
      id: 'B',
      sizeBytes: 600,
      lastUsed: 10,
      buffer: new ArrayBuffer(0),
      unload: vi.fn(),
    };

    manager.registerModel(instanceA);
    const success = await manager.requestLoad('B', 600);
    expect(success).toBe(true);
    expect(instanceA.unload).toHaveBeenCalled(); // Evicted A
  });
});
