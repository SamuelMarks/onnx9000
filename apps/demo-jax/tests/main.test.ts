import { describe, it, expect, vi } from 'vitest';
import { initJaxDemo } from '../src/main.js';

vi.mock('@onnx9000/converters', () => ({
  parseJaxpr: vi.fn().mockReturnValue({ eqns: [{ primitive: 'add' }] }),
}));

describe('demo-jax', () => {
  it('should convert jaxpr', async () => {
    document.body.innerHTML = '<button id="convert-btn"></button><div id="jax-output"></div>';
    initJaxDemo();
    document.getElementById('convert-btn')?.click();
    await new Promise((r) => setTimeout(r, 10));
    expect(document.getElementById('jax-output')?.innerText).toContain('Success! JAX');
  });
});
