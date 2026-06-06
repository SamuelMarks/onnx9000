// @ts-nocheck
import { describe, expect, it, vi } from 'vitest';
import { Editor } from '../src/components/Editor.js';

vi.mock('monaco-editor', () => ({
  editor: {
    create: vi.fn().mockReturnValue({
      setValue: vi.fn(),
      getValue: vi.fn().mockReturnValue('mock code'),
      onDidChangeModelContent: vi.fn().mockReturnValue({ dispose: vi.fn() }),
      layout: vi.fn(),
      dispose: vi.fn(),
      setModel: vi.fn()
    }),
    createModel: vi.fn().mockReturnValue({
      getValue: vi.fn().mockReturnValue('mock content'),
      setValue: vi.fn(),
      dispose: vi.fn()
    }),
    setTheme: vi.fn()
  },
  Uri: { parse: vi.fn() }
}));

global.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
} as any;

describe('Editor', () => {
  it('should render and open files', () => {
    const editor = new Editor({ initialValue: 'test' });
    editor.mount(document.body);

    expect(editor.element.className).toContain('demo-editor-container');

    editor.openFile('test.txt', 'hello');
    expect(editor.getValue()).toBe('mock code');
  });
});
