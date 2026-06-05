import { describe, it, expect } from 'vitest';
import { initOnnxScriptDemo } from '../src/main.js';

describe('demo-onnx-script', () => {
  it('should evaluate script', () => {
    document.body.innerHTML = `
      <button id="runBtn"></button>
      <textarea id="scriptInput">return new Graph("test");</textarea>
      <div id="output"></div>
    `;
    initOnnxScriptDemo();
    document.getElementById('runBtn')?.click();
    expect(document.getElementById('output')?.textContent).toContain(
      'Success! Generated Graph JSON',
    );
  });
});
