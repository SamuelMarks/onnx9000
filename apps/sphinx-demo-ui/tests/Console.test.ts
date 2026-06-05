// @ts-nocheck
import { describe, it, expect } from 'vitest';
import { Console } from '../src/components/Console.js';
import { globalEventBus } from '../src/core/EventBus.js';

describe('Console', () => {
  it('should render and handle logs', () => {
    const consoleComp = new Console();
    document.body.appendChild(consoleComp.element);

    globalEventBus.emit('CONSOLE_LOG', {
      level: 'info',
      message: 'test msg',
      timestamp: new Date()
    });

    expect(consoleComp.element.innerHTML).toContain('test msg');

    const clearBtn = consoleComp.element.querySelector(
      '.demo-console-clear-btn'
    ) as HTMLButtonElement;
    clearBtn.click();

    expect(consoleComp.element.innerHTML).not.toContain('test msg');
  });
});
