/**
 * @fileoverview BottomContainer.ts
 * Provides BottomContainer functionality for the Sphinx Demo UI.
 */
import { Component } from '../core/Component';
import { globalEventBus } from '../core/EventBus';
import { t } from '../core/I18n';
import { Logger } from '../core/Logger';
import { Console } from './Console';
import { OnnxVisualizer } from './OnnxVisualizer';
import { Tabs } from './Tabs';

/** Represents the `BottomContainer` class. */
export class BottomContainer extends Component<HTMLDivElement> {
  /** Property `tabs` of `BottomContainer`. */
  private tabs!: Tabs;
  /** Property `_boundLanguageChanged` of `BottomContainer`. */
  private _boundLanguageChanged: () => void;

  /** Initializes a new `BottomContainer` instance. */
  constructor() {
    super();
    this._boundLanguageChanged = () => this.updateI18n();
    this.element = this.render();
  }

  /**
   * Executes `render` on `BottomContainer`.
   * @returns Result of `render`.
   */
  protected render(): HTMLDivElement {
    const container = document.createElement('div');
    container.className = 'demo-pane-bottom';
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.height = '100%';
    container.style.flex = '1';

    // Create tab panels
    const consolePanel = document.createElement('div');
    consolePanel.id = 'demo-console-panel';
    consolePanel.style.height = '100%';

    const consoleComp = new Console();
    consoleComp.mount(consolePanel);

    // Start intercepting logs so the UI captures them
    Logger.getInstance().startIntercepting();

    const vizPanel = document.createElement('div');
    vizPanel.id = 'demo-viz-panel';
    vizPanel.style.height = '100%';
    const vizComp = new OnnxVisualizer();
    vizComp.mount(vizPanel);

    this.tabs = new Tabs({
      tabs: [
        { id: 'console', label: t('bottom.console'), content: consolePanel },
        { id: 'viz', label: t('bottom.visualizer'), content: vizPanel },
      ],
      initialTabId: 'console',
      onChange: (tabId) => {
        console.log('Tab switched to:', tabId);
        globalEventBus.emit('TAB_CHANGED', tabId);
      },
    });

    this.tabs.mount(container);

    return container;
  }

  /**
   * Executes `updateI18n` on `BottomContainer`.
   * @returns Result of `updateI18n`.
   */
  private updateI18n() {
    if (this.tabs) {
      const tabBtnConsole = this.element.querySelector('#tab-console');
      const tabBtnViz = this.element.querySelector('#tab-viz');
      if (tabBtnConsole) tabBtnConsole.textContent = t('bottom.console');
      if (tabBtnViz) tabBtnViz.textContent = t('bottom.visualizer');
    }
  }

  /** Executes `onMount` on `BottomContainer`. */
  protected onMount(): void {
    globalEventBus.on('LANGUAGE_CHANGED', this._boundLanguageChanged);
    this.onCleanup(() => {
      globalEventBus.off('LANGUAGE_CHANGED', this._boundLanguageChanged);
    });
  }
}
