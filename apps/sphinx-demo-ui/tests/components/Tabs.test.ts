import { describe, it, expect, vi } from 'vitest';
import { Tabs } from '../../src/components/Tabs';

describe('Tabs', () => {
  const c1 = document.createElement('div');
  c1.textContent = 'Content 1';
  const c2 = document.createElement('div');
  c2.textContent = 'Content 2';
  const c3 = document.createElement('div');
  c3.textContent = 'Content 3';

  const mockTabs = [
    { id: 't1', label: 'Tab 1', content: c1 },
    { id: 't2', label: 'Tab 2', content: c2 },
    { id: 't3', label: 'Tab 3', content: c3 }
  ];

  it('should render and set initial active tab', () => {
    const tabs = new Tabs({ tabs: mockTabs });
    const el = (tabs as any).element as HTMLElement;

    expect(el.className).toBe('demo-tabs-container');
    const activeBtn = el.querySelector('button.active');
    expect(activeBtn).not.toBeNull();
    expect(activeBtn?.textContent).toBe('Tab 1');

    const visiblePanel = el.querySelector('#panel-t1') as HTMLElement;
    expect(visiblePanel.style.display).toBe('block');
    expect(visiblePanel.textContent).toContain('Content 1');

    const hiddenPanel = el.querySelector('#panel-t2') as HTMLElement;
    expect(hiddenPanel.style.display).toBe('none');
  });

  it('should respect initialTabId option', () => {
    const tabs = new Tabs({ tabs: mockTabs, initialTabId: 't2' });
    const el = (tabs as any).element as HTMLElement;

    const activeBtn = el.querySelector('button.active');
    expect(activeBtn?.textContent).toBe('Tab 2');

    expect(tabs.getActiveTabId()).toBe('t2');
  });

  it('should switch tabs on click', () => {
    const changeSpy = vi.fn();
    const tabs = new Tabs({ tabs: mockTabs, onChange: changeSpy });
    const el = (tabs as any).element as HTMLElement;
    tabs.mount(document.body);

    const btn2 = el.querySelector('#tab-t2') as HTMLButtonElement;
    btn2.click();

    expect(tabs.getActiveTabId()).toBe('t2');
    expect(changeSpy).toHaveBeenCalledWith('t2');

    const panel2 = el.querySelector('#panel-t2') as HTMLElement;
    expect(panel2.style.display).toBe('block');

    const panel1 = el.querySelector('#panel-t1') as HTMLElement;
    expect(panel1.style.display).toBe('none');

    // click again does nothing
    btn2.click();
    expect(changeSpy).toHaveBeenCalledTimes(1); // not called again
  });

  it('should navigate via keyboard arrows', () => {
    const tabs = new Tabs({ tabs: mockTabs });
    const el = (tabs as any).element as HTMLElement;
    tabs.mount(document.body);

    const tabList = el.querySelector('.demo-tab-list') as HTMLElement;

    // Right arrow -> tab 2
    tabList.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight' }));
    expect(tabs.getActiveTabId()).toBe('t2');

    // Right arrow -> tab 3
    tabList.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight' }));
    expect(tabs.getActiveTabId()).toBe('t3');

    // Right arrow (loop) -> tab 1
    tabList.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight' }));
    expect(tabs.getActiveTabId()).toBe('t1');

    // Left arrow (loop) -> tab 3
    tabList.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft' }));
    expect(tabs.getActiveTabId()).toBe('t3');

    // Left arrow -> tab 2
    tabList.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft' }));
    expect(tabs.getActiveTabId()).toBe('t2');

    // Home -> tab 1
    tabList.dispatchEvent(new KeyboardEvent('keydown', { key: 'Home' }));
    expect(tabs.getActiveTabId()).toBe('t1');

    // End -> tab 3
    tabList.dispatchEvent(new KeyboardEvent('keydown', { key: 'End' }));
    expect(tabs.getActiveTabId()).toBe('t3');

    // Unhandled key does nothing
    tabList.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
    expect(tabs.getActiveTabId()).toBe('t3');
  });

  it('should do nothing on keydown if no tab is active', () => {
    // Weird state but theoretically possible if manipulated
    const tabs = new Tabs({ tabs: mockTabs });
    const el = (tabs as any).element as HTMLElement;

    // remove tabindex=0 to break it
    el.querySelector('[tabindex="0"]')?.setAttribute('tabindex', '-1');

    const tabList = el.querySelector('.demo-tab-list') as HTMLElement;
    tabList.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight' }));

    expect(tabs.getActiveTabId()).toBe('t1');
  });
});

it('should ignore default key handler path when not arrow/home/end', () => {
  const c1 = document.createElement('div');
  const mockTabs2 = [{ id: 't1', label: 'Tab 1', content: c1 }];
  const tabs = new Tabs({ tabs: mockTabs2 });
  const el = (tabs as any).element as HTMLElement;
  tabs.mount(document.body);

  // Setup spy on selectTab
  const selectSpy = vi.spyOn(tabs, 'selectTab');

  const tabList = el.querySelector('.demo-tab-list') as HTMLElement;
  tabList.dispatchEvent(new KeyboardEvent('keydown', { key: 'a' }));

  expect(selectSpy).not.toHaveBeenCalled();
});
