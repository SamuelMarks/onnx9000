import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

describe('demo-compile main', () => {
  let btnCompile: any;
  let outEl: any;

  beforeEach(() => {
    btnCompile = { addEventListener: vi.fn() };
    outEl = { textContent: '' };

    vi.stubGlobal('document', {
      getElementById: vi.fn((id) => {
        if (id === 'btn-compile') return btnCompile;
        if (id === 'output') return outEl;
        return null;
      }),
    });

    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.runOnlyPendingTimers();
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.resetModules();
  });

  it('runs compile demo successfully', async () => {
    await import('../src/main.js');
    expect(btnCompile.addEventListener).toHaveBeenCalledWith('click', expect.any(Function));

    const clickHandler = btnCompile.addEventListener.mock.calls[0][1];
    clickHandler();

    expect(outEl.textContent).toBe('Compiling...\n');
    vi.advanceTimersByTime(500);

    expect(outEl.textContent).toContain('[OK] AOT Compilation finished: model.bin');
  });

  it('handles missing elements', async () => {
    vi.mocked(document.getElementById).mockReturnValue(null);
    await import('../src/main.js');
  });
});
