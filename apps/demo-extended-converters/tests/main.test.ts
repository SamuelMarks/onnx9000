import { describe, it, expect, vi } from 'vitest';
import { initExtendedConvertersDemo } from '../src/main.js';
import { mmdnn } from '@onnx9000/converters';

vi.mock('@onnx9000/converters', () => ({
  mmdnn: {
    convert: vi.fn().mockResolvedValue('Mocked Result Payload'),
  },
}));

describe('demo-extended-converters', () => {
  it('should handle conversion', async () => {
    document.body.innerHTML = `
      <button id="btnConvert"></button>
      <div id="output"></div>
      <input type="file" id="fileInput" />
      <select id="srcFramework"><option value="keras">Keras</option></select>
      <select id="dstFramework"><option value="onnx">ONNX</option></select>
    `;
    initExtendedConvertersDemo();
    const btn = document.getElementById('btnConvert') as HTMLButtonElement;

    // Check no files first
    btn.click();
    expect(document.getElementById('output')?.textContent).toBe(
      'Please select one or more files to convert.',
    );

    // Mock files
    const fileInput = document.getElementById('fileInput') as HTMLInputElement;
    Object.defineProperty(fileInput, 'files', {
      value: [new File([''], 'model.h5')],
    });

    btn.click();
    await new Promise((r) => setTimeout(r, 10)); // flush promises

    expect(document.getElementById('output')?.textContent).toContain('Conversion Successful!');
    expect(document.getElementById('output')?.textContent).toContain('Result Type: String Payload');
  });
});
