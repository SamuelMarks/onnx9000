import { Router } from './router';
import { safeJsonParse } from './middleware';

export interface KServeInput {
  name: string;
  shape: number[];
  datatype: string;
  data?: ReturnType<typeof JSON.parse>[];
  parameters?: Record<string, ReturnType<typeof JSON.parse>>;
}

export interface KServeOutput {
  name: string;
  parameters?: Record<string, ReturnType<typeof JSON.parse>>;
}

export interface KServeRequest {
  id?: string;
  parameters?: Record<string, ReturnType<typeof JSON.parse>>;
  inputs: KServeInput[];
  outputs?: KServeOutput[];
}

export interface KServeResponseOutput {
  name: string;
  shape: number[];
  datatype: string;
  data: ReturnType<typeof JSON.parse>[];
  parameters?: Record<string, ReturnType<typeof JSON.parse>>;
}

export interface KServeResponse {
  model_name: string;
  model_version?: string;
  id?: string;
  parameters?: Record<string, ReturnType<typeof JSON.parse>>;
  outputs: KServeResponseOutput[];
}

const VALID_DATATYPES = new Set([
  'BOOL',
  'UINT8',
  'UINT16',
  'UINT32',
  'UINT64',
  'INT8',
  'INT16',
  'INT32',
  'INT64',
  'FP16',
  'FP32',
  'FP64',
  'BYTES',
]);

function validateKServeRequest(body: ReturnType<typeof JSON.parse>): KServeRequest {
  if (!body.inputs || !Array.isArray(body.inputs)) {
    throw new Error('KServe request must contain an "inputs" array');
  }

  for (const input of body.inputs) {
    if (!input.name || typeof input.name !== 'string') {
      throw new Error('Input missing or invalid "name"');
    }
    if (!input.shape || !Array.isArray(input.shape)) {
      throw new Error(`Input ${input.name} missing or invalid "shape" array`);
    }
    if (!input.datatype || !VALID_DATATYPES.has(input.datatype)) {
      throw new Error(`Input ${input.name} has invalid datatype: ${input.datatype}`);
    }
    if (input.data && Array.isArray(input.data)) {
      const expectedLength = input.shape.reduce((a: number, b: number) => a * b, 1);
      if (expectedLength > 0 && input.data.length !== expectedLength) {
        throw new Error(
          `Input ${input.name} data length (${input.data.length}) does not match shape (${expectedLength})`,
        );
      }
    }
  }
  return body as KServeRequest;
}

import { Tensor } from '@onnx9000/core';
import { Onnx9000Server } from './index';
export function addKServeRoutes(server: Onnx9000Server, router: Router) {
  router.get('/v2/health/ready', async () => {
    return new Response(JSON.stringify({ ready: true }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  });

  router.get('/v2/health/live', async () => {
    return new Response(JSON.stringify({ live: true }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  });

  router.get('/v2/models', async () => {
    return new Response(JSON.stringify({ models: [] }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  });

  router.get('/v2/models/:model_name', async (req, params) => {
    return new Response(
      JSON.stringify({
        name: params.model_name,
        versions: ['1'],
        platform: 'onnx9000',
        inputs: [],
        outputs: [],
        parameters: {
          execution_provider: 'wasm',
          memory_usage_bytes: 1024,
        },
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      },
    );
  });

  router.get('/v2/models/:model_name/versions/:version', async (req, params) => {
    return new Response(
      JSON.stringify({
        name: params.model_name,
        versions: [params.version],
        platform: 'onnx9000',
        inputs: [],
        outputs: [],
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      },
    );
  });

  router.post('/v2/models/:model_name/infer', async (req, params) => {
    const contentType = req.headers.get('content-type') || '';

    if (
      contentType.includes('application/octet-stream') ||
      req.headers.has('Inference-Header-Content-Length')
    ) {
      const buffer = await req.arrayBuffer();
      const headerLengthStr = req.headers.get('Inference-Header-Content-Length');

      if (headerLengthStr) {
        const headerLength = parseInt(headerLengthStr, 10);
        const jsonBuffer = buffer.slice(0, headerLength);
        const decoder = new TextDecoder('utf-8');
        const jsonStr = decoder.decode(jsonBuffer);
        const kserveReq = validateKServeRequest(safeJsonParse(jsonStr));

        const isLittleEndian = true;
        if (kserveReq.parameters && kserveReq.parameters['endianness'] === 'big') {
          // would byteswap here
        }

        const response: KServeResponse = {
          model_name: params.model_name || '',
          model_version: '1',
          id: kserveReq.id || '1',
          parameters: {},
          outputs: kserveReq.inputs.map((input) => ({
            name: input.name,
            datatype: input.datatype,
            shape: input.shape,
            data: [],
          })),
        };

        const outJson = JSON.stringify(response);
        const outJsonBuffer = new TextEncoder().encode(outJson);
        const outDataBuffer = buffer.slice(headerLength);

        const combined = new Uint8Array(outJsonBuffer.length + outDataBuffer.byteLength);
        combined.set(outJsonBuffer, 0);
        combined.set(new Uint8Array(outDataBuffer), outJsonBuffer.length);

        return new Response(combined, {
          status: 200,
          headers: {
            'Content-Type': 'application/octet-stream',
            'Inference-Header-Content-Length': outJsonBuffer.length.toString(),
          },
        });
      }

      return new Response(buffer, {
        status: 200,
        headers: {
          'Content-Type': 'application/octet-stream',
        },
      });
    }

    if (contentType.includes('multipart/form-data')) {
      const formData = await req.formData();
      const files: string[] = [];
      for (const [key, value] of formData.entries()) {
        if (value instanceof File) {
          files.push(value.name);
        }
      }
      return new Response(JSON.stringify({ uploaded_files: files }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    try {
      const rawText = await req.text();
      const rawBody = safeJsonParse(rawText);
      const kserveReq = validateKServeRequest(rawBody);

      // Execute the graph logic natively
      const inputTensors: Record<string, Tensor> = {};
      for (const input of kserveReq.inputs) {
        inputTensors[input.name] = new Tensor(
          input.name,
          input.shape,
          input.datatype === 'FP32' ? 'float32' : 'int32', // simplified mapping
          false,
          true,
          input.data && input.datatype === 'FP32'
            ? new Float32Array(input.data)
            : new Int32Array(input.data || []),
        );
      }

      // Simulate session execution (in reality we would lookup model and run session.run)
      const outputTensors = inputTensors; // Identity fallback without loaded model

      const response: KServeResponse = {
        model_name: params.model_name || '',
        model_version: '1',
        id: kserveReq.id || '1',
        parameters: {},
        outputs: Object.values(outputTensors).map((t) => ({
          name: t.name || '',
          datatype: t.dtype === 'float32' ? 'FP32' : 'INT32',
          shape: t.shape as number[],
          data: Array.from(t.data as ReturnType<typeof JSON.parse>),
        })),
      };

      return new Response(JSON.stringify(response), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    } catch (_err) {
      const err = _err instanceof Error ? _err : new Error(String(_err));
      return new Response(
        JSON.stringify({
          error: err.message,
          stack: err.stack,
        }),
        {
          status: 400,
          headers: { 'Content-Type': 'application/json' },
        },
      );
    }
  });
}
