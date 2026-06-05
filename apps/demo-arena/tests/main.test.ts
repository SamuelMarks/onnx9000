import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

describe('demo-arena main', () => {
  let btnProfiler: any;
  let btnRefresh: any;
  let peakMem: any;
  let blocksContainer: any;

  beforeEach(() => {
    btnProfiler = { addEventListener: vi.fn() };
    btnRefresh = { addEventListener: vi.fn() };
    peakMem = { textContent: '' };
    blocksContainer = { innerHTML: '', appendChild: vi.fn() };

    vi.stubGlobal('document', {
      getElementById: vi.fn((id) => {
        if (id === 'run-profiler') return btnProfiler;
        if (id === 'refresh-arena') return btnRefresh;
        if (id === 'peak-mem') return peakMem;
        if (id === 'blocks') return blocksContainer;
        return null;
      }),
      createElement: vi.fn(() => ({ className: '', textContent: '' })),
    });

    vi.stubGlobal('Math', {
      ...Math,
      random: vi.fn(() => 0.5),
      floor: Math.floor,
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.resetModules();
  });

  it('binds events and runs profiler', async () => {
    await import('../src/main.js');
    expect(btnProfiler.addEventListener).toHaveBeenCalledWith('click', expect.any(Function));

    const clickHandler = btnProfiler.addEventListener.mock.calls[0][1];
    clickHandler();
    expect(peakMem.textContent).toBe('100.00'); // 0.5 * 100 + 50 = 100
  });

  it('binds events and refreshes arena', async () => {
    await import('../src/main.js');
    expect(btnRefresh.addEventListener).toHaveBeenCalledWith('click', expect.any(Function));

    const clickHandler = btnRefresh.addEventListener.mock.calls[0][1];
    clickHandler();
    expect(blocksContainer.innerHTML).toBe('');
    expect(blocksContainer.appendChild).toHaveBeenCalled(); // 0.5 * 10 + 5 = 10 blocks
    expect(blocksContainer.appendChild).toHaveBeenCalledTimes(10);
  });

  it('handles null elements safely', async () => {
    vi.mocked(document.getElementById).mockReturnValue(null);
    await import('../src/main.js');

    // Trigger branch logic where peakMem and blocksContainer are null
    if (btnProfiler.addEventListener.mock.calls.length > 0) {
      btnProfiler.addEventListener.mock.calls[0][1]();
    }
    if (btnRefresh.addEventListener.mock.calls.length > 0) {
      btnRefresh.addEventListener.mock.calls[0][1]();
    }
  });
});
