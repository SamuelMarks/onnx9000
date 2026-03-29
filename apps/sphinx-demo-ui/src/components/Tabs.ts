/* eslint-disable */
// @ts-nocheck
import { Component } from '../core/Component';

export interface TabItem {
  id: string;
  label: string;
  content: HTMLElement;
}

export interface TabsOptions {
  tabs: TabItem[];
  initialTabId?: string;
  onChange?: (tabId: string) => void;
}

/**
 * A highly accessible vanilla JS Tabs component.
 * Supports keyboard navigation (Left/Right arrows, Home, End).
 */
export class Tabs extends Component<HTMLDivElement> {
  private options: TabsOptions;
  private activeTabId: string | null = null;
  private tabButtons: Map<string, HTMLButtonElement> = new Map();
  private tabPanels: Map<string, HTMLElement> = new Map();

  constructor(options: TabsOptions) {
    super();
    this.options = options;

    if (this.options.tabs.length > 0) {
      this.activeTabId = this.options.initialTabId || this.options.tabs[0].id;
    }

    this.element = this.render();
  }

  protected render(): HTMLDivElement {
    const container = document.createElement('div');
    container.className = 'demo-tabs-container';
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.height = '100%';
    container.style.width = '100%';
    container.style.flex = '1';

    const tabList = document.createElement('div');
    tabList.className = 'demo-tab-list';
    tabList.setAttribute('role', 'tablist');

    const panelContainer = document.createElement('div');
    panelContainer.className = 'demo-tab-panels';
    panelContainer.style.flex = '1';
    panelContainer.style.overflow = 'auto';
    panelContainer.style.position = 'relative';

    this.options.tabs.forEach((tab, index) => {
      const isSelected = tab.id === this.activeTabId;

      // Button
      const btn = document.createElement('button');
      btn.className = 'demo-tab-button';
      if (isSelected) btn.classList.add('active');
      btn.setAttribute('role', 'tab');
      btn.setAttribute('aria-selected', isSelected.toString());
      btn.setAttribute('aria-controls', `panel-${tab.id}`);
      btn.setAttribute('id', `tab-${tab.id}`);
      btn.setAttribute('tabindex', isSelected ? '0' : '-1');
      btn.setAttribute('data-index', index.toString());
      btn.textContent = tab.label;

      this.tabButtons.set(tab.id, btn);
      tabList.appendChild(btn);

      // Panel Wrapper
      const panel = document.createElement('div');
      panel.className = 'demo-tab-panel';
      panel.setAttribute('role', 'tabpanel');
      panel.setAttribute('id', `panel-${tab.id}`);
      panel.setAttribute('aria-labelledby', `tab-${tab.id}`);
      panel.style.display = isSelected ? 'block' : 'none';
      panel.style.height = '100%';

      panel.appendChild(tab.content);
      this.tabPanels.set(tab.id, panel);
      panelContainer.appendChild(panel);
    });

    container.appendChild(tabList);
    container.appendChild(panelContainer);

    return container;
  }

  protected onMount(): void {
    // Click handlers
    this.tabButtons.forEach((btn, tabId) => {
      this.addDOMListener(btn, 'click', () => this.selectTab(tabId));
    });

    // Keyboard handlers on the tablist
    const tabList = this.element.querySelector('.demo-tab-list') as HTMLElement;
    this.addDOMListener(tabList, 'keydown', (e) => this.handleKeyDown(e as KeyboardEvent));
  }

  private handleKeyDown(e: KeyboardEvent): void {
    const activeBtn = Array.from(this.tabButtons.values()).find(
      (b) => b.getAttribute('tabindex') === '0'
    );
    if (!activeBtn) return;

    let currentIndex = parseInt(activeBtn.getAttribute('data-index') || '0', 10);
    const maxIndex = this.options.tabs.length - 1;

    let newIndex = currentIndex;

    switch (e.key) {
      case 'ArrowLeft':
        e.preventDefault();
        newIndex = currentIndex > 0 ? currentIndex - 1 : maxIndex;
        break;
      case 'ArrowRight':
        e.preventDefault();
        newIndex = currentIndex < maxIndex ? currentIndex + 1 : 0;
        break;
      case 'Home':
        e.preventDefault();
        newIndex = 0;
        break;
      case 'End':
        e.preventDefault();
        newIndex = maxIndex;
        break;
      default:
        return;
    }

    if (newIndex !== currentIndex) {
      const targetTab = this.options.tabs[newIndex];
      this.selectTab(targetTab.id);
      this.tabButtons.get(targetTab.id)?.focus();
    }
  }

  public selectTab(tabId: string): void {
    if (this.activeTabId === tabId) return;

    // Deactivate current
    if (this.activeTabId) {
      const oldBtn = this.tabButtons.get(this.activeTabId);
      const oldPanel = this.tabPanels.get(this.activeTabId);
      if (oldBtn) {
        oldBtn.classList.remove('active');
        oldBtn.setAttribute('aria-selected', 'false');
        oldBtn.setAttribute('tabindex', '-1');
      }
      if (oldPanel) {
        oldPanel.style.display = 'none';
      }
    }

    // Activate new
    this.activeTabId = tabId;
    const newBtn = this.tabButtons.get(tabId);
    const newPanel = this.tabPanels.get(tabId);

    if (newBtn) {
      newBtn.classList.add('active');
      newBtn.setAttribute('aria-selected', 'true');
      newBtn.setAttribute('tabindex', '0');
    }
    if (newPanel) {
      newPanel.style.display = 'block';
    }

    if (this.options.onChange) {
      this.options.onChange(tabId);
    }
  }

  public getActiveTabId(): string | null {
    return this.activeTabId;
  }
}
