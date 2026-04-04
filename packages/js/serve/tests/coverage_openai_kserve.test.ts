import { describe, it, expect } from 'vitest';
import { createServer } from '../src/index';

describe('Coverage OpenAI & KServe', () => {
  const server = createServer();

  it('OpenAI - /v1/chat/completions normal', async () => {
    const req = new Request('http://localhost/v1/chat/completions', {
      method: 'POST',
      body: JSON.stringify({
        messages: [{ role: 'user', content: 'test' }],
        model: 'test-model',
        max_tokens: 10,
        temperature: 0.5,
        top_p: 0.9,
        top_k: 40,
        stop: ['\n'],
        tools: [],
        presence_penalty: 0.1,
      }),
    });
    const res = await server.fetch(req);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.choices[0].message.content).toContain('test-model');
  });

  it('OpenAI - /v1/completions', async () => {
    const req = new Request('http://localhost/v1/completions', {
      method: 'POST',
      body: JSON.stringify({ model: 'test-model' }),
    });
    const res = await server.fetch(req);
    expect(res.status).toBe(200);
  });

  it('OpenAI - /v1/embeddings float', async () => {
    const req = new Request('http://localhost/v1/embeddings', {
      method: 'POST',
      body: JSON.stringify({ model: 'test-model', encoding_format: 'float' }),
    });
    const res = await server.fetch(req);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.data[0].embedding).toEqual([0, 0.1, 0.2]);
  });

  it('OpenAI - /v1/embeddings base64', async () => {
    const req = new Request('http://localhost/v1/embeddings', {
      method: 'POST',
      body: JSON.stringify({ model: 'test-model', encoding_format: 'base64' }),
    });
    const res = await server.fetch(req);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(typeof body.data[0].embedding).toBe('string');
  });

  it('OpenAI - /v1/audio/transcriptions', async () => {
    let req = new Request('http://localhost/v1/audio/transcriptions', { method: 'POST' });
    let res = await server.fetch(req);
    expect(res.status).toBe(400); // missing multipart

    const formData = new FormData();
    formData.append('file', new Blob(['test'], { type: 'audio/wav' }), 'test.wav');
    req = new Request('http://localhost/v1/audio/transcriptions', {
      method: 'POST',
      body: formData,
    });
    res = await server.fetch(req);
    expect(res.status).toBe(200);
  });

  it('KServe - health', async () => {
    expect((await server.fetch(new Request('http://localhost/v2/health/ready'))).status).toBe(200);
    expect((await server.fetch(new Request('http://localhost/v2/health/live'))).status).toBe(200);
  });

  it('KServe - models', async () => {
    expect((await server.fetch(new Request('http://localhost/v2/models'))).status).toBe(200);
    expect((await server.fetch(new Request('http://localhost/v2/models/m1'))).status).toBe(200);
    expect(
      (await server.fetch(new Request('http://localhost/v2/models/m1/versions/1'))).status,
    ).toBe(200);
  });

  it('KServe - infer JSON', async () => {
    // Bad request
    const badReq1 = new Request('http://localhost/v2/models/m1/infer', {
      method: 'POST',
      body: '{}',
    });
    let res = await server.fetch(badReq1);
    expect(res.status).toBe(400);

    const badReq2 = new Request('http://localhost/v2/models/m1/infer', {
      method: 'POST',
      body: '{"inputs":[{}]}',
    });
    res = await server.fetch(badReq2);
    expect(res.status).toBe(400); // missing name

    const badReq3 = new Request('http://localhost/v2/models/m1/infer', {
      method: 'POST',
      body: '{"inputs":[{"name":"test"}]}',
    });
    res = await server.fetch(badReq3);
    expect(res.status).toBe(400); // missing shape

    const badReq4 = new Request('http://localhost/v2/models/m1/infer', {
      method: 'POST',
      body: '{"inputs":[{"name":"test", "shape":[1]}]}',
    });
    res = await server.fetch(badReq4);
    expect(res.status).toBe(400); // missing datatype

    const badReq5 = new Request('http://localhost/v2/models/m1/infer', {
      method: 'POST',
      body: '{"inputs":[{"name":"test", "shape":[1], "datatype":"FP32", "data":[1,2]}]}',
    });
    res = await server.fetch(badReq5);
    expect(res.status).toBe(400); // mismatch shape and data length

    // Good request INT32
    const goodReq = new Request('http://localhost/v2/models/m1/infer', {
      method: 'POST',
      body: '{"id":"test-id", "inputs":[{"name":"test", "shape":[1], "datatype":"INT32", "data":[1]}]}',
    });
    res = await server.fetch(goodReq);
    expect(res.status).toBe(200);
  });

  it('KServe - infer Binary / octet-stream', async () => {
    // octet stream without header length
    let req = new Request('http://localhost/v2/models/m1/infer', {
      method: 'POST',
      headers: { 'content-type': 'application/octet-stream' },
      body: new Uint8Array([1, 2, 3]),
    });
    let res = await server.fetch(req);
    expect(res.status).toBe(200);

    // octet stream WITH header length
    const jsonStr =
      '{"id":"b-1", "parameters":{"endianness":"big"}, "inputs":[{"name":"test", "shape":[1], "datatype":"FP32", "data":[1]}]}';
    const encoder = new TextEncoder();
    const jsonBuf = encoder.encode(jsonStr);

    const combined = new Uint8Array(jsonBuf.length + 4);
    combined.set(jsonBuf, 0);
    combined.set(new Uint8Array([0, 0, 0, 0]), jsonBuf.length);

    req = new Request('http://localhost/v2/models/m1/infer', {
      method: 'POST',
      headers: {
        'content-type': 'application/octet-stream',
        'Inference-Header-Content-Length': jsonBuf.length.toString(),
      },
      body: combined,
    });
    res = await server.fetch(req);
    expect(res.status).toBe(200);
  });

  it('KServe - infer multipart', async () => {
    const formData = new FormData();
    formData.append('file', new Blob(['test'], { type: 'audio/wav' }), 'test.wav');
    formData.append('notfile', 'test');
    const req = new Request('http://localhost/v2/models/m1/infer', {
      method: 'POST',
      body: formData,
    });
    const res = await server.fetch(req);
    expect(res.status).toBe(200);
  });
});
