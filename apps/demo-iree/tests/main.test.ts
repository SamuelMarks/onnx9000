import { describe, it, expect, vi } from 'vitest';
import { initIreeDemo } from '../src/main.js';

vi.mock('@onnx9000/iree-compiler/src/cli.js', () => ({
  compileModel: vi.fn().mockResolvedValue(undefined),
}));
vi.mock('@onnx9000/iree-runtime/src/vm.js', () => ({
  Module: class {},
  Context: class {},
  WVMInterpreter: class {
    runAsync = vi.fn().mockResolvedValue(undefined);
  },
  HALBindings: { register: vi.fn() },
}));

describe('demo-iree', () => {
  it('should compile and run', async () => {
    document.body.innerHTML = `
      <button id="compile-btn"></button>
      <button id="run-btn" disabled></button>
      <div id="compiler-output"></div>
      <div id="runtime-output"></div>
    `;
    initIreeDemo();
    document.getElementById('compile-btn')?.click();
    await new Promise((r) => setTimeout(r, 10));
    expect(document.getElementById('compiler-output')?.innerText).toContain(
      'Compilation successful',
    );

    document.getElementById('run-btn')?.disabled = false;
    document.getElementById('run-btn')?.click();
    await new Promise((r) => setTimeout(r, 10));
    expect(document.getElementById('runtime-output')?.innerText).toContain('Execution successful');
  });
});
