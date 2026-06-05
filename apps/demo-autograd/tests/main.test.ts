import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

describe('demo-autograd main', () => {
  let gradBtn: any;
  let outEl: any;

  beforeEach(() => {
    gradBtn = { addEventListener: vi.fn(), disabled: false };
    outEl = { innerText: '' };

    vi.stubGlobal('document', {
      getElementById: vi.fn((id) => {
        if (id === 'grad-btn') return gradBtn;
        if (id === 'autograd-output') return outEl;
        return null;
      }),
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.resetModules();
  });

  it('runs autograd demo successfully', async () => {
    await import('../src/main.js');
    expect(gradBtn.addEventListener).toHaveBeenCalledWith('click', expect.any(Function));

    const clickHandler = gradBtn.addEventListener.mock.calls[0][1];
    const runPromise = clickHandler();
    expect(outEl.innerText).toContain('Initializing Autograd Engine...');

    await runPromise;
    expect(outEl.innerText).toContain(
      'Success! Augmented ONNX graph now computes forward pass + gradients.',
    );
    expect(gradBtn.disabled).toBe(true);
  });

  it('handles error gracefully', async () => {
    await import('../src/main.js');
    const clickHandler = gradBtn.addEventListener.mock.calls[0][1];

    // forcefully mock setTimeout to throw
    vi.stubGlobal(
      'setTimeout',
      vi.fn(() => {
        throw new Error('Timeout failed');
      }),
    );

    await clickHandler();
    expect(outEl.innerText).toContain('Error: Timeout failed');
  });
});
