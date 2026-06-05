// @ts-nocheck
import { describe, it, expect, vi } from 'vitest';
import { initDemoUI } from '../src/main.js';
import { WasmManager } from '../src/core/WasmManager.js';

vi.mock('../src/core/WasmManager.js', () => ({
  WasmState: { LOADED: 'loaded' },
  WasmManager: {
    getInstance: vi.fn().mockReturnValue({ state: 'loaded' })
  }
}));

// We need a dummy DOM container
describe('main', () => {
  it('should initialize UI', () => {
    const container = document.createElement('div');
    container.id = 'interactive-demo-container';
    document.body.appendChild(container);

    initDemoUI('interactive-demo-container');

    expect(container.innerHTML).toContain('demo-ui-root');
  });
});
