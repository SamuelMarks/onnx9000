// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initOnnxScriptDemo } from '../src/main.js';

describe('demo-onnx-script', () => {
  beforeEach(() => {
    document.body.innerHTML =
      '<textarea id="scriptInput"></textarea><button id="runBtn"></button><div id="output"></div>';
  });

  it('should execute valid and invalid ONNX scripts', () => {
    document.body.innerHTML = '';
    initOnnxScriptDemo();

    document.body.innerHTML =
      '<textarea id="scriptInput"></textarea><button id="runBtn"></button><div id="output"></div>';
    initOnnxScriptDemo();

    const input = document.getElementById('scriptInput') as HTMLTextAreaElement;
    const btn = document.getElementById('runBtn') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLDivElement;

    input.value = 'return new Graph("scripted");';
    btn.click();
    expect(out.textContent).toContain('Success! Generated Graph JSON:');

    input.value = 'const err = new Error("Syntax failure"); throw err;';
    btn.click();
    expect(out.textContent).toContain('Error executing script: Syntax failure');

    input.value = 'throw "string failure";';
    btn.click();
    expect(out.textContent).toContain('Error executing script: string failure');
  });
});
