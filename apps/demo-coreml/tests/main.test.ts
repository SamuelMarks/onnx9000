// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { coremlJsonReplacer, initCoremlDemo } from '../src/main.js';

describe('demo-coreml', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="convert-btn"></button><div id="output"></div>';
  });

  it('should convert ONNX graph to CoreML MIL AST and format bigint values', () => {
    expect(coremlJsonReplacer('k', 42n)).toBe('42n');
    expect(coremlJsonReplacer('k', 'str')).toBe('str');

    document.body.innerHTML = '';
    initCoremlDemo();

    document.body.innerHTML = '<button id="convert-btn"></button><div id="output"></div>';
    initCoremlDemo();

    const btn = document.getElementById('convert-btn') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLElement;

    btn.click();
    expect(out.innerText).toContain('main');
  });
});
