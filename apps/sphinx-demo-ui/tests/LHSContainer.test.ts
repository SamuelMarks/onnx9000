// @ts-nocheck
import { describe, expect, it, vi } from 'vitest';
import { LHSContainer } from '../src/components/LHSContainer.js';
import { globalEventBus } from '../src/core/EventBus.js';

vi.mock('monaco-editor', () => ({
  editor: {
    create: vi.fn().mockReturnValue({
      setValue: vi.fn(),
      getValue: vi.fn().mockReturnValue('{}'),
      onDidChangeModelContent: vi.fn().mockReturnValue({ dispose: vi.fn() }),
      layout: vi.fn(),
      dispose: vi.fn(),
      setModel: vi.fn(),
    }),
    createModel: vi.fn().mockReturnValue({
      getValue: vi.fn().mockReturnValue('mock content'),
      setValue: vi.fn(),
      dispose: vi.fn(),
    }),
  },
  Uri: { parse: vi.fn() },
}));

global.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
} as any;

describe('LHSContainer', () => {
  it('should render and emit events', async () => {
    const lhs = new LHSContainer();
    document.body.appendChild(lhs.element);

    let eventFired = false;
    globalEventBus.on('CONVERSION_STARTED', () => {
      eventFired = true;
    });

    const runBtn = lhs.element.querySelector('.demo-btn-run-conversion') as HTMLButtonElement;
    runBtn.click();

    await new Promise((r) => setTimeout(r, 10));
    expect(eventFired).toBe(true);
  });
});
