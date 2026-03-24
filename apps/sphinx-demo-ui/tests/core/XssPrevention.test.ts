import { describe, it, expect, vi } from 'vitest';
import { Console } from '../../src/components/Console';
import { LogLevel } from '../../src/core/Logger';
import { OnnxVisualizer } from '../../src/components/OnnxVisualizer';

// We must mock Cytoscape for JSDOM otherwise it crashes due to lack of canvas API
const mockCyInstance = {
  on: vi.fn(),
  destroy: vi.fn(),
  elements: vi.fn(() => ({ remove: vi.fn() })),
  add: vi.fn(),
  layout: vi.fn(() => ({ run: vi.fn() })),
  fit: vi.fn()
};

vi.mock('cytoscape', () => {
  return {
    default: vi.fn(() => mockCyInstance)
  };
});

describe('XSS Prevention', () => {
  it('Console should not execute malicious script tags in messages', () => {
    const consoleComp = new Console();
    consoleComp.mount(document.body);
    const el = (consoleComp as any).element as HTMLElement;
    const outputDiv = el.querySelector('.demo-console-output') as HTMLDivElement;

    const xssPayload = '<script>window.__XSS_FLAG__ = true;</script>';

    consoleComp.appendLog({
      level: LogLevel.INFO,
      message: xssPayload,
      timestamp: new Date()
    });

    const msgSpan = outputDiv.querySelector('.demo-console-msg') as HTMLElement;
    // Uses textContent, which escapes HTML inherently
    expect(msgSpan.innerHTML).toBe('&lt;script&gt;window.__XSS_FLAG__ = true;&lt;/script&gt;');
    expect((window as any).__XSS_FLAG__).toBeUndefined();

    consoleComp.unmount();
  });

  it('OnnxVisualizer should not execute malicious scripts in tooltips', () => {
    const viz = new OnnxVisualizer();
    viz.mount(document.body);

    // Simulate node tap with malicious payload
    const onCalls = mockCyInstance.on.mock.calls;
    const nodeTapCb = onCalls.find((c: any) => c[0] === 'tap' && c[1] === 'node')[2];

    const maliciousData = {
      type: 'operator',
      label: '<img src="x" onerror="window.__XSS_FLAG_2__=true;">',
      attributes: {
        '<script>alert(1)</script>': 'val'
      }
    };

    const mockEvt = {
      target: { data: () => maliciousData },
      renderedPosition: { x: 0, y: 0 }
    };

    nodeTapCb(mockEvt);

    // The tooltip uses innerHTML for formatting. We ensure DOM logic isn't triggered
    expect((window as any).__XSS_FLAG_2__).toBeUndefined();
    viz.unmount();
  });
});
