// @ts-nocheck

import * as monaco from 'monaco-editor';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Editor } from '../../src/components/Editor';
import { globalEventBus } from '../../src/core/EventBus';

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
