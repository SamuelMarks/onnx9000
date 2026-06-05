import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

describe('demo-array main', () => {
  let runBtn: any;
  let outEl: any;

  beforeEach(() => {
    runBtn = { addEventListener: vi.fn(), disabled: false };
    outEl = { innerText: '' };

    vi.stubGlobal('document', {
      getElementById: vi.fn((id) => {
        if (id === 'run-btn') return runBtn;
        if (id === 'array-output') return outEl;
        return null;
      }),
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.resetModules();
  });

  it('runs array api demo successfully', async () => {
    // we need to mock @onnx9000/array
    vi.doMock('@onnx9000/array', () => ({
      array: vi.fn((arr) => ({ numpy: () => arr, opType: 'ArrayNode' })),
      add: vi.fn((a, b) => {
        if (!a.numpy() || !a.numpy()[2]) return { numpy: () => undefined, opType: 'Add' };
        return {
          numpy: () => [
            a.numpy()[0] + b.numpy()[0],
            a.numpy()[1] + b.numpy()[1],
            a.numpy()[2] + b.numpy()[2],
          ],
          opType: 'Add',
        };
      }),
      matmul: vi.fn(() => ({ numpy: () => undefined, opType: 'MatMul' })),
      lazy_mode: vi.fn(),
    }));

    await import('../src/main.js');
    expect(runBtn.addEventListener).toHaveBeenCalledWith('click', expect.any(Function));

    const clickHandler = runBtn.addEventListener.mock.calls[0][1];

    const runPromise = clickHandler();
    await runPromise;
    expect(outEl.innerText).toContain('Initializing Web-Native Array API');
    expect(outEl.innerText).toContain('Success! The Array API is fully functional.');
    expect(runBtn.disabled).toBe(false);
  });

  it('handles errors gracefully', async () => {
    vi.doMock('@onnx9000/array', () => ({
      array: vi.fn(() => {
        throw new Error('Array init failed');
      }),
    }));

    await import('../src/main.js');
    const clickHandler = runBtn.addEventListener.mock.calls[0][1];

    await clickHandler();
    expect(outEl.innerText).toContain('Error: Array init failed');
    expect(runBtn.disabled).toBe(false);
  });
});
