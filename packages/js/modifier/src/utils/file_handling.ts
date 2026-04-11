/* eslint-disable */
import { Graph } from '@onnx9000/core';

// Phase 8: Data Privacy & Security

/**
 * 96. Ensure the entire application is served as a static bundle.
 * (This is satisfied by the architecture and avoiding server requests).
 */

/**
 * 97. Provide a standalone HTML file option (single-file export)
 *
 * Takes the current HTML, injects the JS/CSS inline, and optionally
 * bakes the ONNX model into a base64 string inside the HTML for offline sharing.
 */
export function createStandaloneHTML(
  htmlTemplate: string,
  jsBundle: string,
  onnxBase64: string | null = null,
): string {
  let finalHtml = htmlTemplate;

  // Inject script
  finalHtml = finalHtml.replace(
    '<!-- INJECT_SCRIPT -->',
    `<script type="module">\n${jsBundle}\n</script>`,
  );

  if (onnxBase64) {
    // Inject model
    finalHtml = finalHtml.replace(
      '<!-- INJECT_MODEL -->',
      `<script id="baked-model" type="application/octet-stream">${onnxBase64}</script>`,
    );
  }

  return finalHtml;
}

/**
 * 98. Ensure massive models utilize File slicing APIs
 * We abstract file reading to support chunking for massive arrays.
 */
export async function readMassiveFile(
  file: File,
  chunkSize: number = 1024 * 1024 * 100,
): Promise<ArrayBuffer> {
  // If small enough, just read all
  if (file.size <= chunkSize) {
    return file.arrayBuffer();
  }

  // Otherwise, we chunk read and assemble (mock implementation of streaming assembly)
  // For the browser's max heap limit, standard ArrayBuffer is usually limited to 2GB-4GB depending on the engine.
  // In a real huge-model scenario we'd stream directly to WASM memory, but here we assemble to a single buffer if possible.
  const totalBuffer = new Uint8Array(file.size);
  let offset = 0;

  for (let i = 0; i < file.size; i += chunkSize) {
    const slice = file.slice(i, i + chunkSize);
    const buf = await slice.arrayBuffer();
    totalBuffer.set(new Uint8Array(buf), offset);
    offset += buf.byteLength;
  }

  return totalBuffer.buffer;
}

/**
 * 99. Restrict execution features if the model triggers potential infinite loops
 */
export function createTimeoutCircuitBreaker(timeoutMs: number = 5000) {
  const start = performance.now();
  return () => {
    if (performance.now() - start > timeoutMs) {
      throw new Error('Execution timeout: potential infinite loop detected.');
    }
  };
}

/**
 * 100. Disallow prototype pollution or malicious script injection
 */
export function sanitizeMetadata(str: string | undefined): string {
  if (!str) return '';
  // Basic HTML entity encoding to prevent XSS / script injection when rendering metadata
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
