/* eslint-disable */
// @ts-nocheck
import { describe, it, expect, beforeEach } from 'vitest';
import { Console } from '../../src/components/Console';
import { globalEventBus } from '../../src/core/EventBus';
import { LogEntry, LogLevel } from '../../src/core/Logger';

describe('Console', () => {
  beforeEach(() => {
    globalEventBus.clearAll();
  });

  it('should render and mount components', () => {
    const consoleComp = new Console();
    const el = (consoleComp as object).element as HTMLElement;

    expect(el.className).toBe('demo-console-container');
    expect(el.querySelector('.demo-console-toolbar')).not.toBeNull();
    expect(el.querySelector('.demo-console-output')).not.toBeNull();
    expect(el.querySelector('.demo-console-clear-btn')).not.toBeNull();
  });

  it('should clear logs when clear button is clicked', () => {
    const consoleComp = new Console();
    consoleComp.mount(document.body);
    const el = (consoleComp as object).element as HTMLElement;

    const outputDiv = el.querySelector('.demo-console-output') as HTMLDivElement;
    outputDiv.innerHTML = '<div class="demo-console-line">test</div>';

    expect(outputDiv.children.length).toBe(1);

    const clearBtn = el.querySelector('.demo-console-clear-btn') as HTMLButtonElement;
    clearBtn.click();

    expect(outputDiv.children.length).toBe(0);

    consoleComp.unmount();
  });

  it('should append log on CONSOLE_LOG event', () => {
    const consoleComp = new Console();
    consoleComp.mount(document.body);
    const el = (consoleComp as object).element as HTMLElement;
    const outputDiv = el.querySelector('.demo-console-output') as HTMLDivElement;

    const entry: LogEntry = {
      level: LogLevel.INFO,
      message: 'Hello Console',
      timestamp: new Date('2026-03-23T10:00:00.000Z') // mock date
    };

    globalEventBus.emit('CONSOLE_LOG', entry);

    expect(outputDiv.children.length).toBe(1);

    const line = outputDiv.children[0] as HTMLElement;
    expect(line.className).toContain('demo-console-level-info');

    const msg = line.querySelector('.demo-console-msg');
    expect(msg?.textContent).toBe('Hello Console');

    const time = line.querySelector('.demo-console-time');
    expect(time?.textContent).toContain('10:00:00.000');

    consoleComp.unmount();
  });

  it('should handle log overflow (max 1000 lines)', () => {
    const consoleComp = new Console();
    consoleComp.mount(document.body);
    const el = (consoleComp as object).element as HTMLElement;
    const outputDiv = el.querySelector('.demo-console-output') as HTMLDivElement;

    const baseEntry: LogEntry = {
      level: LogLevel.INFO,
      message: 'test',
      timestamp: new Date()
    };

    // Add 1005 lines
    for (let i = 0; i < 1005; i++) {
      consoleComp.appendLog({ ...baseEntry, message: `test ${i}` });
    }

    // Should be truncated to 1000
    expect(outputDiv.children.length).toBe(1000);
    // The first message should be test 5 since 0-4 were removed
    expect(outputDiv.children[0].querySelector('.demo-console-msg')?.textContent).toBe('test 5');

    consoleComp.unmount();
  });
});
