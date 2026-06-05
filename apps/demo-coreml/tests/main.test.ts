import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

describe('demo-coreml main', () => {
  let convertBtn: any;
  let outEl: any;

  beforeEach(() => {
    convertBtn = { addEventListener: vi.fn() };
    outEl = { innerText: '' };

    vi.stubGlobal('document', {
      getElementById: vi.fn((id) => {
        if (id === 'convert-btn') return convertBtn;
        if (id === 'output') return outEl;
        return null;
      }),
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.resetModules();
  });

  it('runs coreml conversion successfully', async () => {
    vi.doMock('@onnx9000/core', () => {
      return {
        Graph: class {
          inputs = [];
          outputs = [];
          nodes = [];
          constructor(public name: string) {}
        },
        Node: class {
          inputs = [];
          outputs = [];
          constructor(public opType: string) {}
        },
      };
    });

    vi.doMock('@onnx9000/coreml', () => ({
      convertToCoreML: vi.fn(() => ({ result: 'ast', val: 10n })),
    }));

    await import('../src/main.js');
    expect(convertBtn.addEventListener).toHaveBeenCalledWith('click', expect.any(Function));

    const clickHandler = convertBtn.addEventListener.mock.calls[0][1];
    clickHandler();

    expect(outEl.innerText).toContain('10n');
    expect(outEl.innerText).toContain('"result": "ast"');
  });

  it('handles errors gracefully', async () => {
    vi.doMock('@onnx9000/core', () => {
      return {
        Graph: class {
          inputs = [];
          outputs = [];
          nodes = [];
        },
        Node: class {},
      };
    });
    vi.doMock('@onnx9000/coreml', () => ({
      convertToCoreML: vi.fn(() => {
        throw new Error('MIL Conversion Failed');
      }),
    }));

    await import('../src/main.js');
    const clickHandler = convertBtn.addEventListener.mock.calls[0][1];

    clickHandler();
    expect(outEl.innerText).toContain('Error: MIL Conversion Failed');
  });
});
