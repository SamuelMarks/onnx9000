import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Editor } from '../../src/components/Editor';
import { globalEventBus } from '../../src/core/EventBus';
import * as monaco from 'monaco-editor';

describe('Editor Theme Sync', () => {
  beforeEach(() => {
    globalEventBus.clearAll();
    vi.clearAllMocks();
  });

  it('should listen to THEME_CHANGED globally', () => {
    const editor = new Editor();
    editor.mount(document.body);

    globalEventBus.emit('THEME_CHANGED', 'vs-dark');

    expect(monaco.editor.setTheme).toHaveBeenCalledWith('vs-dark');

    globalEventBus.emit('THEME_CHANGED', 'vs-light');

    expect(monaco.editor.setTheme).toHaveBeenCalledWith('vs-light');

    editor.unmount();
  });
});
