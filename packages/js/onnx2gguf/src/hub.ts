/**
 * @fileoverview hub.ts
 * Provides hub functionality for the onnx2gguf package.
 */
import { Onnx9000Error } from '@onnx9000/core';
/**
 * Executes `fetchHfConfig`.
 * @param repoId Parameter `repoId`.
 * @param token Parameter `token`.
 * @returns Result of `fetchHfConfig`.
 */
export async function fetchHfConfig(
  repoId: string,
  token?: string,
): Promise<{
  config: ReturnType<typeof JSON.parse>;
  tokenizer: string;
  url: string;
}> {
  const headers: Record<string, string> = {};
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  const url = `https://huggingface.co/${repoId}/resolve/main`;

  let config = {};
  let tokenizer = '';

  try {
    const configRes = await fetch(`${url}/config.json`, { headers });
    if (!configRes.ok) {
      if (configRes.status === 404) throw new Onnx9000Error('config.json not found (404)');

      if (configRes.status === 403 || configRes.status === 401)
        throw new Onnx9000Error('Unauthorized (403/401) - Check your token');
    } else {
      config = await configRes.json();
    }
  } catch (_e) {
    const e = _e instanceof Error ? _e : new Error(String(_e));
    console.warn('Failed to fetch config:', e.message);
  }

  try {
    const tokRes = await fetch(`${url}/tokenizer.json`, { headers });
    if (tokRes.ok) {
      tokenizer = await tokRes.text();
    }
  } catch (_e) {
    const e = _e instanceof Error ? _e : new Error(String(_e));

    console.warn('Failed to fetch tokenizer:', e.message);
  }

  return { config, tokenizer, url: `https://huggingface.co/${repoId}` };
}

/**
 * Executes `generateReadme`.
 * @param modelName Parameter `modelName`.
 * @param originalRepo Parameter `originalRepo`.
 * @param quantization Parameter `quantization`.
 * @returns Result of `generateReadme`.
 */
export function generateReadme(
  modelName: string,
  originalRepo: string,
  quantization: string,
): string {
  return `---
base_model: ${originalRepo}
tags:
- gguf
- onnx9000
---

# ${modelName} GGUF

This model was exported to GGUF format from ${originalRepo} using the [onnx9000](https://github.com/samuel/onnx9000) zero-dependency compiler.

## Quantization
- **Level:** ${quantization}
- **Format:** GGUF v3

## Usage
\`\`\`bash
llama-cli -m ${modelName.toLowerCase()}-${quantization.toLowerCase()}.gguf -p "Hello, world!"
\`\`\`
`;
}
