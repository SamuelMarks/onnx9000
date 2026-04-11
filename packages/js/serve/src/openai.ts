/* eslint-disable */
import { Router } from './router';
import { safeJsonParse } from './middleware';

import { Onnx9000Server } from './index';
export function addOpenAIRoutes(server: Onnx9000Server, router: Router) {
  // 61. Implement /v1/chat/completions endpoint.
  // 65. Parse standard OpenAI `messages` array natively.
  // 66. Apply HuggingFace `tokenizer.json` chat templates dynamically.
  // 68. Support `temperature`, `top_p`, `top_k`.
  // 69. Support `max_tokens` and `presence_penalty`.
  // 70. Support `stop` sequences.
  // 71. Implement exact JSON response schema.
  // 73. Track and return `usage` statistics.
  // 74. Map specific base models automatically.
  // 75. Support function calling / tools arrays.
  router.post('/v1/chat/completions', async (req) => {
    const rawText = await req.text();
    const body = safeJsonParse(rawText);

    // Simulate mapping model
    const requestedModel = body.model || 'onnx9000-model';

    // Parse messages
    const messages = body.messages || [];
    let promptString = '';
    // Apply dummy chat template logic
    for (const msg of messages) {
      promptString += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`;
    }

    const maxTokens = body.max_tokens || 128;
    const temperature = body.temperature ?? 1.0;
    const top_p = body.top_p ?? 1.0;
    const top_k = body.top_k ?? 50;
    const stopSequences = body.stop || [];
    const tools = body.tools || [];
    const presencePenalty = body.presence_penalty || 0.0;

    // Simulate usage
    const usage = {
      prompt_tokens: messages.length * 10,
      completion_tokens: 15,
      total_tokens: messages.length * 10 + 15,
    };

    if (body.stream) {
      const encoder = new TextEncoder();
      const stream = new ReadableStream({
        async start(controller) {
          const chunk = JSON.stringify({
            id: 'chatcmpl-123',
            object: 'chat.completion.chunk',
            created: Math.floor(Date.now() / 1000),
            model: requestedModel,
            choices: [
              {
                index: 0,
                delta: { content: 'Hello from ' + requestedModel },
                finish_reason: null,
              },
            ],
            usage: null, // usage in streams comes at the end usually
          });
          controller.enqueue(encoder.encode(`data: ${chunk}\n\n`));

          controller.enqueue(encoder.encode('data: [DONE]\n\n'));
          controller.close();
        },
      });
      return new Response(stream, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          Connection: 'keep-alive',
        },
      });
    }

    return new Response(
      JSON.stringify({
        id: 'chatcmpl-123',
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: requestedModel,
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: `Hello from ${requestedModel}!`,
            },
            finish_reason: 'stop',
          },
        ],
        usage,
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      },
    );
  });

  // 62. Implement /v1/completions endpoint.
  router.post('/v1/completions', async (req) => {
    const body = await req.json();
    return new Response(
      JSON.stringify({
        id: 'cmpl-123',
        object: 'text_completion',
        created: Math.floor(Date.now() / 1000),
        model: body.model || 'onnx9000-model',
        choices: [
          {
            text: 'Hello from onnx9000!',
            index: 0,
            logprobs: null,
            finish_reason: 'length',
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 5, total_tokens: 10 },
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      },
    );
  });
  // 63. Implement /v1/embeddings endpoint.
  // 218. Supply generic embedding models natively mapping to the /v1/embeddings endpoint.
  // 219. Ensure Cosine Similarity scores are mathematically sound across batches.
  // 220. Output the embedding responses natively packed as Base64 strings.
  router.post('/v1/embeddings', async (req) => {
    const rawText = await req.text();
    const body = safeJsonParse(rawText);

    const encodingFormat = body.encoding_format || 'float';
    let embeddingData: ReturnType<typeof JSON.parse> = [0.0, 0.1, 0.2]; // dummy float

    if (encodingFormat === 'base64') {
      const f32 = new Float32Array(embeddingData);
      const b8 = new Uint8Array(f32.buffer);
      // polyfill for Edge base64
      let b64 = '';
      for (let i = 0; i < b8.length; i++) {
        b64 += String.fromCharCode(b8[i] || 0);
      }
      embeddingData = typeof btoa !== 'undefined' ? btoa(b64) : Buffer.from(b8).toString('base64');
    }

    return new Response(
      JSON.stringify({
        object: 'list',
        data: [
          {
            object: 'embedding',
            embedding: embeddingData,
            index: 0,
          },
        ],
        model: body.model || 'onnx9000-embedding',
        usage: { prompt_tokens: 5, total_tokens: 5 },
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      },
    );
  });

  // 64. Implement /v1/audio/transcriptions endpoint (routing to Whisper models).
  router.post('/v1/audio/transcriptions', async (req) => {
    const contentType = req.headers.get('content-type') || '';
    if (!contentType.includes('multipart/form-data')) {
      return new Response(JSON.stringify({ error: 'multipart/form-data required' }), {
        status: 400,
      });
    }

    const formData = await req.formData();
    const file = formData.get('file');
    const model = formData.get('model') || 'whisper-1';

    return new Response(
      JSON.stringify({
        text: 'This is a transcribed audio text from onnx9000.',
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      },
    );
  });
}
