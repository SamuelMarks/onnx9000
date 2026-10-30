// @vitest-environment jsdom
import * as ireeCli from '@onnx9000/iree-compiler';
import * as ireeVm from '@onnx9000/iree-runtime';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { initIreeDemo } from '../src/main.js';

describe('demo-iree', () => {
  beforeEach(() => {
    document.body.innerHTML =
      '<button id="compile-btn"></button><button id="run-btn"></button><div id="compile-output"></div><div id="run-output"></div>';
  });

  it('should compile and run IREE WebVM bytecode and handle errors', async () => {
    document.body.innerHTML = '';
    initIreeDemo();

    document.body.innerHTML =
      '<button id="compile-btn"></button><button id="run-btn"></button><div id="compile-output"></div><div id="run-output"></div>';
    initIreeDemo();

    const compileBtn = document.getElementById('compile-btn') as HTMLButtonElement;
    const runBtn = document.getElementById('run-btn') as HTMLButtonElement;
    const compileOut = document.getElementById('compile-output') as HTMLElement;
    const runOut = document.getElementById('run-output') as HTMLElement;

    runBtn.click();

    compileBtn.click();
    await new Promise((r) => setTimeout(r, 30));
    expect(compileOut.innerText).toContain('Compilation successful.');

    runBtn.click();
    await new Promise((r) => setTimeout(r, 30));
    expect(runOut.innerText).toContain('Execution successful!');

    const compileSpy = vi
      .spyOn(ireeCli, 'compileModel')
      .mockRejectedValueOnce(new Error('Bad MLIR'));
    compileBtn.click();
    await new Promise((r) => setTimeout(r, 30));
    expect(compileOut.innerText).toContain('Compiler Error: Bad MLIR');

    compileSpy.mockRejectedValueOnce('Raw compile err');
    compileBtn.click();
    await new Promise((r) => setTimeout(r, 30));
    expect(compileOut.innerText).toContain('Compiler Error: Raw compile err');
    compileSpy.mockRestore();

    const runSpy = vi
      .spyOn(ireeVm.WVMInterpreter.prototype, 'runAsync')
      .mockRejectedValueOnce(new Error('VM Trap'));
    runBtn.click();
    await new Promise((r) => setTimeout(r, 30));
    expect(runOut.innerText).toContain('Runtime Error: VM Trap');

    runSpy.mockRejectedValueOnce('Raw VM Trap');
    runBtn.click();
    await new Promise((r) => setTimeout(r, 30));
    expect(runOut.innerText).toContain('Runtime Error: Raw VM Trap');
    runSpy.mockRestore();
  });
});
