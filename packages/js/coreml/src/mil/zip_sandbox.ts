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
export function validateZipInputData(files: Map<string, Uint8Array>): void {
  // 281. Sandbox the JSZip/Archive generation to ensure no cross-site scripting attacks via malicious model metadata.
  // We inspect Manifest.json specifically.
  const manifestData = files.get('Manifest.json');
  if (manifestData) {
    const text = new TextDecoder().decode(manifestData);
    if (text.includes('<script>') || text.includes('javascript:')) {
      throw new Error('Security Violation: Malicious payload detected inside model metadata.');
    }
  }

  for (const [key, _] of files.entries()) {
    // 282. Prevent local file-system access vulnerabilities (directory traversal attacks)
    if (key.includes('../') || key.startsWith('/')) {
      throw new Error(`Security Violation: Directory traversal detected in package path: ${key}`);
    }
  }
}
