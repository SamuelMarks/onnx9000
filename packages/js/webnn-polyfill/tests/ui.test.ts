/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { PolyfillUI } from '../src/ui.js';

describe('PolyfillUI', () => {
  let ui: PolyfillUI;

  beforeEach(() => {
    document.body.innerHTML = '';
    ui = new PolyfillUI();
  });

  it('should initialize and attach to body', () => {
    expect(document.getElementById('webnn-polyfill-ui-container')).not.toBeNull();
  });

  it('should show and hide spinner', () => {
    const hide = ui.showSpinner('Testing...');
    expect(document.body.textContent).toContain('[WebNN] ⌛ Testing...');
    hide();
    expect(document.body.textContent).not.toContain('[WebNN] ⌛ Testing...');
  });

  it('should show flamegraph and auto-hide', async () => {
    vi.useFakeTimers();
    ui.showFlamegraph([{ name: 'test-op', time: 10 }]);
    expect(document.body.innerHTML).toContain('WebNN Flamegraph');
    expect(document.body.innerHTML).toContain('test-op');

    vi.advanceTimersByTime(5001);
    expect(document.body.innerHTML).not.toContain('WebNN Flamegraph');
    vi.useRealTimers();
  });

  it('should log graph info to console', () => {
    const spy = vi.spyOn(console, 'log').mockImplementation(() => {});
    ui.logGraphConsoleUI({
      nodes: [{ opType: 'Relu', name: 'r1', inputs: ['in'], outputs: ['out'] }],
    });
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });

  it('should show and close AST visualizer', () => {
    ui.showAST({
      nodes: [{ opType: 'Add', name: 'a1', inputs: ['a', 'b'], outputs: ['c'] }],
    });
    expect(document.body.innerHTML).toContain('ONNX AST Visualizer');

    const closeBtn = document.querySelector('button');
    expect(closeBtn).not.toBeNull();
    closeBtn?.click();
    expect(document.body.innerHTML).not.toContain('ONNX AST Visualizer');
  });

  it('should have attached to window via index.ts', async () => {
    // Reload index to trigger side effects
    await import('../src/index.js?t=' + Date.now());
    expect((window.navigator as Navigator & { ml?: object }).ml).toBeDefined();
    expect((window as Window & { MLContext?: object }).MLContext).toBeDefined();
    expect((window as Window & { MLGraphBuilder?: object }).MLGraphBuilder).toBeDefined();
  });
});
