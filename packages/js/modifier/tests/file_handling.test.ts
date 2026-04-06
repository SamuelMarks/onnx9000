import { describe, it, expect, vi } from 'vitest';
import {
  createStandaloneHTML,
  readMassiveFile,
  createTimeoutCircuitBreaker,
  sanitizeMetadata,
} from '../src/utils/file_handling.js';

describe('Phase 8: Data Privacy & Security', () => {
  it('97. createStandaloneHTML injects content correctly', () => {
    const template = `<html><body><!-- INJECT_SCRIPT --><!-- INJECT_MODEL --></body></html>`;

    // No model
    const out1 = createStandaloneHTML(template, 'console.log("hello");');
    expect(out1).toContain('<script type="module">\nconsole.log("hello");\n</script>');
    expect(out1).toContain('<!-- INJECT_MODEL -->');

    // With model
    const out2 = createStandaloneHTML(template, 'console.log("hello");', 'base64str');
    expect(out2).toContain(
      '<script id="baked-model" type="application/octet-stream">base64str</script>',
    );
  });

  it('98. readMassiveFile chunk-reads a large Mock File', async () => {
    // Mock a global File API to simulate reading
    const file = {
      size: 300,
      slice: (start: number, end: number) => {
        return {
          arrayBuffer: async () => new Uint8Array(Math.min(end - start, 300 - start)).buffer,
        };
      },
      arrayBuffer: async () => new Uint8Array(300).buffer,
    } as Object as File;

    const fullBuf = await readMassiveFile(file, 100);
    expect(fullBuf.byteLength).toBe(300);

    const smallFile = {
      size: 50,
      arrayBuffer: async () => new Uint8Array(50).buffer,
    } as Object as File;

    const smallBuf = await readMassiveFile(smallFile, 100);
    expect(smallBuf.byteLength).toBe(50);
  });

  it('99. createTimeoutCircuitBreaker works', () => {
    // 1ms timeout
    const breaker = createTimeoutCircuitBreaker(1);

    // mock time passage
    const start = performance.now();
    while (performance.now() - start < 5) {
      // wait
    }

    // should throw
    expect(() => breaker()).toThrow('Execution timeout: potential infinite loop detected.');
  });

  it('100. sanitizeMetadata prevents XSS', () => {
    expect(sanitizeMetadata(undefined)).toBe('');
    expect(sanitizeMetadata('<script>alert("xss")</script>')).toBe(
      '&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;',
    );
    expect(sanitizeMetadata('hello & welcome')).toBe('hello &amp; welcome');
    expect(sanitizeMetadata("'single'")).toBe('&#039;single&#039;');
  });
});
