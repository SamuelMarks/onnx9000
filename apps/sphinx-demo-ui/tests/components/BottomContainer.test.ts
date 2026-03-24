import { describe, it, expect, vi, beforeEach } from 'vitest';
import { BottomContainer } from '../../src/components/BottomContainer';
import { Logger } from '../../src/core/Logger';

// Mock child components
vi.mock('../../src/components/Tabs', () => ({
  Tabs: class {
    options: any;
    constructor(options: any) {
      this.options = options;
    }
    mount(el: any) {
      el.appendChild(document.createElement('div'));
    }
    triggerChange(tabId: string) {
      if (this.options.onChange) this.options.onChange(tabId);
    }
  }
}));

vi.mock('../../src/components/Console', () => ({
  Console: class {
    mount(_el: any) {}
  }
}));

vi.mock('../../src/components/OnnxVisualizer', () => ({
  OnnxVisualizer: class {
    mount(_el: any) {}
  }
}));

describe('BottomContainer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should render and mount components', () => {
    const loggerSpy = vi
      .spyOn(Logger.getInstance(), 'startIntercepting')
      .mockImplementation(() => {});

    const container = new BottomContainer();
    const el = (container as any).element as HTMLElement;

    expect(el.className).toBe('demo-pane-bottom');
    expect(loggerSpy).toHaveBeenCalled();
  });

  it('should trigger onChange callback in Tabs', () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});

    const container = new BottomContainer();
    const tabs = (container as any).tabs;

    tabs.triggerChange('viz');

    expect(consoleSpy).toHaveBeenCalledWith('Tab switched to:', 'viz');
    consoleSpy.mockRestore();
  });
});
