import { describe, it, expect } from 'vitest';
import { validateZipInputData } from '../src/index.js';

describe('JSZip Sandboxing', () => {
  it('Validates clean structure without errors', () => {
    const files = new Map<string, Uint8Array>();
    files.set('Manifest.json', new TextEncoder().encode(JSON.stringify({ test: 123 })));
    files.set('Data/com.apple.CoreML/model.mlmodel', new Uint8Array([1, 2, 3]));
    expect(() => validateZipInputData(files)).not.toThrow();
  });

  it('Detects and blocks script injections in Manifest.json', () => {
    const files = new Map<string, Uint8Array>();
    files.set(
      'Manifest.json',
      new TextEncoder().encode(JSON.stringify({ author: '<script>alert(1)</script>' })),
    );
    expect(() => validateZipInputData(files)).toThrowError(/Malicious payload detected/);

    files.set(
      'Manifest.json',
      new TextEncoder().encode(JSON.stringify({ author: 'javascript:evil()' })),
    );
    expect(() => validateZipInputData(files)).toThrowError(/Malicious payload detected/);
  });

  it('Detects and blocks path traversal attempts', () => {
    const files = new Map<string, Uint8Array>();
    files.set('Manifest.json', new TextEncoder().encode(JSON.stringify({ test: 123 })));
    files.set('../../../etc/passwd', new Uint8Array());
    expect(() => validateZipInputData(files)).toThrowError(/Directory traversal detected/);

    const files2 = new Map<string, Uint8Array>();
    files2.set('/absolute/path/write.bin', new Uint8Array());
    expect(() => validateZipInputData(files2)).toThrowError(/Directory traversal detected/);
  });
});
