// @ts-nocheck
import { describe, it, expect, vi } from 'vitest';
import { RHSContainer } from '../src/components/RHSContainer.js';
import { globalEventBus } from '../src/core/EventBus.js';

vi.mock('monaco-editor', () => ({
  editor: {
    create: vi.fn().mockReturnValue({
      setValue: vi.fn(),
      getValue: vi.fn().mockReturnValue('{}'),
      onDidChangeModelContent: vi.fn().mockReturnValue({ dispose: vi.fn() }),
      layout: vi.fn(),
      dispose: vi.fn(),
      setModel: vi.fn()
    }),
    createModel: vi.fn().mockReturnValue({
      getValue: vi.fn().mockReturnValue('mock content'),
      setValue: vi.fn(),
      dispose: vi.fn()
    })
  },
  Uri: { parse: vi.fn() }
}));

global.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
} as any;

describe('RHSContainer', () => {
  it('should render and handle events', async () => {
    const rhs = new RHSContainer();
    document.body.appendChild(rhs.element);

    globalEventBus.emit('ONNX_BINARY_GENERATED', new Uint8Array(10));
    await new Promise((r) => setTimeout(r, 10));

    expect(rhs.element.className).toContain('demo-pane-rhs');
  });
});
