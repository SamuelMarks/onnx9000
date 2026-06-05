import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

describe('demo-agent main', () => {
  let runBtn: any;
  let promptEl: any;
  let outEl: any;

  beforeEach(() => {
    runBtn = { addEventListener: vi.fn(), disabled: false };
    promptEl = { value: '' };
    outEl = { innerText: '' };

    vi.stubGlobal('document', {
      getElementById: vi.fn((id) => {
        if (id === 'runBtn') return runBtn;
        if (id === 'prompt') return promptEl;
        if (id === 'output') return outEl;
        return null;
      }),
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.resetModules();
  });

  it('binds click event and executes workflow', async () => {
    await import('../src/main.js');
    expect(runBtn.addEventListener).toHaveBeenCalledWith('click', expect.any(Function));

    const clickHandler = runBtn.addEventListener.mock.calls[0][1];

    // Test empty prompt
    await clickHandler();
    expect(runBtn.disabled).toBe(false);

    // Test successful prompt
    promptEl.value = 'test prompt';
    const runPromise = clickHandler();
    expect(runBtn.disabled).toBe(true);
    expect(outEl.innerText).toBe('Initializing AgentRunner...');

    await runPromise;
    expect(outEl.innerText).toContain('Final Answer: 55');
    expect(runBtn.disabled).toBe(false);
  });
});
