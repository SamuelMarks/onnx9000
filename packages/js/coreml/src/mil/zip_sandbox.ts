/**
 * @fileoverview zip_sandbox module for coreml.
 */
/**
 * Security wrappers explicitly isolating execution during ZIP transformations.
 * @module
 */

/**
 * Sandboxes the generated Manifest and folder structure prior to passing it
 * into `JSZip` to prevent embedded script attacks or illegal path traversals.
 * @param files - The generated file buffer.
 * @throws {Error} if malicious scripts are injected into the manifest.
 */
import { Onnx9000Error } from '@onnx9000/core';
/**
 * Executes `validateZipInputData`.
 * @param files Parameter `files`.
 */
export function validateZipInputData(files: Map<string, Uint8Array>): void {
  // 281. Sandbox the JSZip/Archive generation to ensure no cross-site scripting attacks via malicious model metadata.
  // We inspect Manifest.json specifically.
  const manifestData = files.get('Manifest.json');
  if (manifestData) {
    const text = new TextDecoder().decode(manifestData);
    if (text.includes('<script>') || text.includes('javascript:')) {
      throw new Onnx9000Error(
        'Security Violation: Malicious payload detected inside model metadata.',
      );
    }
  }

  for (const [key, _] of files.entries()) {
    // 282. Prevent local file-system access vulnerabilities (directory traversal attacks)
    if (key.includes('../') || key.startsWith('/')) {
      throw new Onnx9000Error(
        `Security Violation: Directory traversal detected in package path: ${key}`,
      );
    }
  }
}
