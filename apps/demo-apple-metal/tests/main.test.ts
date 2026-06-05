import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

describe('demo-apple-metal main', () => {
  let consoleLogSpy: any;
  let runBtn: any;
  let outEl: any;

  beforeEach(() => {
    consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    runBtn = { addEventListener: vi.fn(), disabled: false };
    outEl = { innerText: '' };
    vi.stubGlobal('document', {
      getElementById: vi.fn((id) => {
        if (id === 'run-btn') return runBtn;
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
    vi.restoreAllMocks();
    vi.resetModules();
  });

  it('runs initialization and main loop', async () => {
    await import('../src/main.js');

    expect(runBtn.addEventListener).toHaveBeenCalledWith('click', expect.any(Function));
    const clickHandler = runBtn.addEventListener.mock.calls[0][1];

    clickHandler();
    expect(outEl.innerText).toBe('Initializing Apple Metal...');

    vi.advanceTimersByTime(500);
    expect(outEl.innerText).toContain('Apple Metal engine loaded');
    expect(outEl.innerText).toContain('SUCCESS');
  });
});
