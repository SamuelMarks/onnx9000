import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Editor } from '../../src/components/Editor';
import * as monaco from 'monaco-editor';

describe('Editor', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should instantiate without throwing errors', () => {
    const editor = new Editor();
    const el = (editor as any).element as HTMLElement;
    expect(el.className).toBe('demo-editor-container');
  });

  it('should initialize Monaco editor on mount', () => {
    const editor = new Editor({ initialValue: 'hello world', language: 'typescript' });
    const parent = document.createElement('div');
    editor.mount(parent);

    expect(monaco.editor.create).toHaveBeenCalled();
    const createArgs = vi.mocked(monaco.editor.create).mock.calls[0];

    // First arg is the DOM element
    expect(createArgs[0]).toBe((editor as any).element);

    // Second arg is options
    expect(createArgs[1]).toMatchObject({
      value: 'hello world',
      language: 'typescript',
      automaticLayout: false
    });

    editor.unmount();
  });

  it('should set up ResizeObserver and dispose correctly on unmount', () => {
    // Setup our ResizeObserver mock to be a spy
    const observeSpy = vi.fn();
    const disconnectSpy = vi.fn();
    global.ResizeObserver = class {
      observe = observeSpy;
      unobserve = vi.fn();
      disconnect = disconnectSpy;
    } as any;

    const editor = new Editor();
    const parent = document.createElement('div');
    editor.mount(parent);

    expect(observeSpy).toHaveBeenCalledWith((editor as any).element);

    const monacoEditorInstance = (editor as any).editor;
    editor.unmount();

    expect(disconnectSpy).toHaveBeenCalled();
    expect(monacoEditorInstance.dispose).toHaveBeenCalled();
  });

  it('should handle openFile caching and updating models', () => {
    const editor = new Editor();
    const parent = document.createElement('div');
    editor.mount(parent);

    // Initial open creates a model
    editor.openFile('test.ts', 'const a = 1;', 'typescript');
    expect(monaco.editor.createModel).toHaveBeenCalledTimes(1);

    const modelInstance = vi.mocked(monaco.editor.createModel).mock.results[0].value;
    const setModelSpy = (editor as any).editor.setModel;
    expect(setModelSpy).toHaveBeenCalledWith(modelInstance);

    // Open same file but with unchanged content -> pulls from cache
    vi.clearAllMocks();
    editor.openFile('test.ts', 'const a = 1;', 'typescript');
    expect(monaco.editor.createModel).not.toHaveBeenCalled(); // Cached!
    expect((editor as any).editor.setModel).toHaveBeenCalledWith(modelInstance);

    // Open same file with DIFFERENT content -> updates value
    vi.clearAllMocks();
    // Re-mock getValue to return old content to simulate out-of-sync
    modelInstance.getValue.mockReturnValue('const a = 1;');

    editor.openFile('test.ts', 'const b = 2;', 'typescript');
    expect(monaco.editor.createModel).not.toHaveBeenCalled(); // Still Cached!
    expect(modelInstance.setValue).toHaveBeenCalledWith('const b = 2;');
    expect((editor as any).editor.setModel).toHaveBeenCalledWith(modelInstance);

    editor.unmount();
  });

  it('should provide value accessors', () => {
    const editor = new Editor();
    editor.mount(document.body);

    const monacoInstance = (editor as any).editor;
    monacoInstance.getValue.mockReturnValue('test content');

    expect(editor.getValue()).toBe('test content');

    editor.setValue('new content');
    expect(monacoInstance.setValue).toHaveBeenCalledWith('new content');

    editor.unmount();
  });

  it('should handle theme changes', () => {
    const editor = new Editor();
    editor.setTheme('vs-dark');
    expect(monaco.editor.setTheme).toHaveBeenCalledWith('vs-dark');
  });

  it('should register onChange callback if provided', () => {
    const onChangeSpy = vi.fn();
    const editor = new Editor({ onChange: onChangeSpy });
    editor.mount(document.body);

    const monacoInstance = (editor as any).editor;
    expect(monacoInstance.onDidChangeModelContent).toHaveBeenCalled();

    // Trigger the callback registered in onDidChangeModelContent
    const registeredCallback = monacoInstance.onDidChangeModelContent.mock.calls[0][0];

    monacoInstance.getValue.mockReturnValue('updated content');
    registeredCallback();

    expect(onChangeSpy).toHaveBeenCalledWith('updated content');

    editor.unmount();
  });

  it('should return empty string for getValue when not mounted', () => {
    const editor = new Editor();
    expect(editor.getValue()).toBe('');
  });

  it('should safely do nothing for setValue when not mounted', () => {
    const editor = new Editor();
    expect(() => editor.setValue('test')).not.toThrow();
  });
});

it('should trigger layout on resize', () => {
  let resizeCallback: any = null;
  global.ResizeObserver = class {
    observe = vi.fn();
    unobserve = vi.fn();
    disconnect = vi.fn();
    constructor(cb: any) {
      resizeCallback = cb;
    }
  } as any;

  const editor = new Editor();
  editor.mount(document.body);

  // Call the captured ResizeObserver callback
  if (resizeCallback) resizeCallback();

  const monacoEditorInstance = (editor as any).editor;
  expect(monacoEditorInstance.layout).toHaveBeenCalled();

  editor.unmount();
});

it('should load initial value from localStorage', () => {
  localStorage.setItem('onnx9000-demo-editor-python', 'cached value');

  const editor = new Editor({ language: 'python', initialValue: 'default' });
  editor.mount(document.body);

  const monacoInstance = (editor as any).editor;
  expect(monacoInstance.setValue).toHaveBeenCalledWith('cached value');

  editor.unmount();
  localStorage.clear();
});

it('should save to localStorage on change', () => {
  const editor = new Editor({ language: 'python', onChange: vi.fn() });
  editor.mount(document.body);

  const monacoInstance = (editor as any).editor;
  const registeredCallback = monacoInstance.onDidChangeModelContent.mock.calls[0][0];

  monacoInstance.getValue.mockReturnValue('new cached code');
  registeredCallback();

  expect(localStorage.getItem('onnx9000-demo-editor-python')).toBe('new cached code');

  editor.unmount();
  localStorage.clear();
});
