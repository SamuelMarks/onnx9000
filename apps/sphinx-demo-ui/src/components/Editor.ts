/**
 * @fileoverview Editor.ts
 * Provides Editor functionality for the Sphinx Demo UI.
 */

import * as monaco from 'monaco-editor';
import { Component } from '../core/Component';
import { globalEventBus } from '../core/EventBus';

(window as never as Record<string, unknown>).monaco = monaco;

import editorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker&inline';
import cssWorker from 'monaco-editor/esm/vs/language/css/css.worker?worker&inline';
import htmlWorker from 'monaco-editor/esm/vs/language/html/html.worker?worker&inline';
import jsonWorker from 'monaco-editor/esm/vs/language/json/json.worker?worker&inline';
import tsWorker from 'monaco-editor/esm/vs/language/typescript/ts.worker?worker&inline';

self.MonacoEnvironment = {
  getWorker(_, label) {
    if (label === 'json') {
      return new jsonWorker();
    }
    if (label === 'css' || label === 'scss' || label === 'less') {
      return new cssWorker();
    }
    if (label === 'html' || label === 'handlebars' || label === 'razor') {
      return new htmlWorker();
    }
    if (label === 'typescript' || label === 'javascript') {
      return new tsWorker();
    }
    return new editorWorker();
  },
};

/** Interface `EditorOptions` definition. */
export interface EditorOptions {
  /** Property `language`. */
  language?: string;
  /** Property `initialValue`. */
  initialValue?: string;
  /** Property `readOnly`. */
  readOnly?: boolean;
  /** Property `theme`. */
  theme?: 'vs-light' | 'vs-dark' | 'hc-black';
  /** Property `onChange`. */
  onChange?: (value: string) => void;
}

/**
 * A wrapper component around Monaco Editor.
 * Includes caching and lifecycle hooks for responsive resizing.
 */
export class Editor extends Component<HTMLDivElement> {
  /** Property `editor` of `Editor`. */
  private editor!: monaco.editor.IStandaloneCodeEditor;
  /** Property `options` of `Editor`. */
  private options: EditorOptions;
  /** Property `cachedModels` of `Editor`. */
  private cachedModels: Map<string, monaco.editor.ITextModel> = new Map();
  /* private currentFileId: string | null = null; */
  /** Property `resizeObserver` of `Editor`. */
  private resizeObserver!: ResizeObserver;

  /**
   * Initializes a new `Editor` instance.
   * @param options Parameter `options`.
   */
  constructor(options?: EditorOptions) {
    super();
    this.options = {
      language: 'typescript',
      initialValue: '',
      readOnly: false,
      theme: 'vs-light',
      ...options,
    };

    // Delay instantiation until mount because Monaco needs an attached DOM node
    this.element = this.render();
  }

  /**
   * Executes `render` on `Editor`.
   * @returns Result of `render`.
   */
  protected render(): HTMLDivElement {
    const container = document.createElement('div');
    container.className = 'demo-editor-container';
    container.style.width = '100%';
    container.style.height = '100%';
    container.style.overflow = 'hidden';
    return container;
  }

  /** Executes `onMount` on `Editor`. */
  protected onMount(): void {
    this.editor = monaco.editor.create(this.element, {
      value: this.options.initialValue,
      language: this.options.language,
      theme: this.options.theme,
      readOnly: this.options.readOnly,
      automaticLayout: false, // We handle it via ResizeObserver for better perf/control
      minimap: { enabled: false },
      scrollBeyondLastLine: false,
      wordWrap: 'on',
    });

    // Try to load cached content from localStorage
    if (this.options.initialValue) {
      const stored = localStorage.getItem(`onnx9000-demo-editor-${this.options.language}`);
      if (stored) {
        this.editor.setValue(stored);
      }
    }

    if (this.options.onChange) {
      // Auto-save to local storage gracefully
      try {
        localStorage.setItem(
          `onnx9000-demo-editor-${this.options.language}`,
          this.editor.getValue(),
        );
      } catch (e) {
        console.warn('Failed to save editor state to localStorage', e);
      }
      const disposable = this.editor.onDidChangeModelContent(() => {
        // Try to load cached content from localStorage
        if (this.options.initialValue) {
          const stored = localStorage.getItem(`onnx9000-demo-editor-${this.options.language}`);
          if (stored) {
            this.editor.setValue(stored);
          }
        }

        if (this.options.onChange) {
          // Auto-save to local storage gracefully
          try {
            localStorage.setItem(
              `onnx9000-demo-editor-${this.options.language}`,
              this.editor.getValue(),
            );
          } catch (e) {
            console.warn('Failed to save editor state to localStorage', e);
          }
          this.options.onChange(this.editor.getValue());
        }
      });
      this.onCleanup(() => disposable.dispose());
    }

    // Handle Resize
    this.resizeObserver = new ResizeObserver(() => {
      // Trigger Monaco layout update
      if (this.editor) {
        this.editor.layout();
      }
    });
    this.resizeObserver.observe(this.element);

    this.onCleanup(
      globalEventBus.on<'vs-light' | 'vs-dark' | 'hc-black'>(
        'THEME_CHANGED',
        (theme: 'vs-light' | 'vs-dark' | 'hc-black') => {
          this.setTheme(theme);
        },
      ),
    );

    this.onCleanup(() => {
      this.resizeObserver.disconnect();
      this.editor.dispose();
      this.cachedModels.forEach((model) => {
        model.dispose();
      });
      this.cachedModels.clear();
    });
  }

  /**
   * Switches the active file. Utilizes Monaco's text model caching
   * to preserve cursor state and scroll position.
   *
   * @param fileId - A unique string path or ID for the file.
   * @param content - The text content of the file.
   * @param language - The monaco language identifier.
   */
  public openFile(fileId: string, content: string, language: string = 'plaintext'): void {
    /* this.currentFileId = fileId; */

    let model = this.cachedModels.get(fileId);
    if (!model) {
      model = monaco.editor.createModel(content, language, monaco.Uri.parse(`file://${fileId}`));
      this.cachedModels.set(fileId, model);
    } else {
      // If content changed externally, update it but preserve state
      if (model.getValue() !== content) {
        model.setValue(content);
      }
    }

    if (this.editor) {
      this.editor.setModel(model);
    }
  }

  /**
   * Executes `getValue` on `Editor`.
   * @returns Result of `getValue`.
   */
  public getValue(): string {
    return this.editor ? this.editor.getValue() : '';
  }

  /**
   * Executes `setValue` on `Editor`.
   * @param value Parameter `value`.
   */
  public setValue(value: string): void {
    if (this.editor) {
      this.editor.setValue(value);
    }
  }

  /**
   * Executes `setTheme` on `Editor`.
   * @param theme Parameter `theme`.
   */
  public setTheme(theme: 'vs-light' | 'vs-dark' | 'hc-black'): void {
    monaco.editor.setTheme(theme);
  }
}
