// @ts-nocheck
import { describe, expect, it } from 'vitest';
import { Tabs } from '../src/components/Tabs.js';

describe('Tabs', () => {
  it('should render and select tab', () => {
    const content1 = document.createElement('div');
    const content2 = document.createElement('div');
    const tabs = new Tabs({
      tabs: [
        { id: '1', label: 'Tab 1', content: content1 },
        { id: '2', label: 'Tab 2', content: content2 }
      ]
    });

    tabs.mount(document.body);
    expect(tabs.getActiveTabId()).toBe('1');

    const btn2 = tabs.element.querySelector('#tab-2') as HTMLElement;
    btn2.click();
    expect(tabs.getActiveTabId()).toBe('2');
  });
});
