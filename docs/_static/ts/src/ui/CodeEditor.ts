import { globalEvents } from '../core/State';
import { BaseComponent } from './BaseComponent';

interface MonacoModel {
  id?: string;
}

interface MonacoEditorInstance {
  getValue(): string;
  setValue(val: string): void;
  getModel(): MonacoModel | null;
  onDidChangeModelContent(cb: () => void): void;
  onDidChangeCursorPosition(
    cb: (e: { position: { lineNumber: number; column: number } }) => void,
  ): void;
  createDecorationsCollection(
    decorations: Array<{
      range: unknown;
      options: { className: string; hoverMessage: { value: string } };
    }>,
  ): void;
}

declare const require: {
  config(cfg: { paths: Record<string, string> }): void;
  (deps: string[], cb: () => void): void;
};
declare const monaco: {
  Range: new (startLine: number, startCol: number, endLine: number, endCol: number) => unknown;
  editor: {
    create(container: HTMLElement, options: Record<string, unknown>): MonacoEditorInstance;
    setTheme(theme: string): void;
    setModelLanguage(model: MonacoModel | null, lang: string): void;
  };
};

export class CodeEditor extends BaseComponent {
  private editorInstance: MonacoEditorInstance | null = null;
  private debounceTimer: ReturnType<typeof setTimeout> | null = null;

  mount(): void {
    // 307. Verify Monaco editor web workers are loaded securely via blob URLs.
    const win = window as Window & {
      MonacoEnvironment?: { getWorkerUrl: (workerId: string, label: string) => string };
    };
    if (typeof win.MonacoEnvironment === 'undefined') {
      win.MonacoEnvironment = {
        getWorkerUrl: (_workerId: string, _label: string) => {
          const proxy = `self.MonacoEnvironment = { baseUrl: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/' }; importScripts('https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs/base/worker/workerMain.js');`;
          return URL.createObjectURL(new Blob([proxy], { type: 'text/javascript' }));
        },
      };
    }

    if (typeof require !== 'undefined') {
      require(['vs/editor/editor.main'], () => {
        this.initEditor();
      });
    } else {
      console.warn('Monaco editor loader not found.');
    }
  }

  private initEditor(): void {
    const isDark = document.body.getAttribute('data-theme') === 'dark';

    this.editorInstance = monaco.editor.create(this.container, {
      value: [
        'import onnxscript',
        'from onnxscript import opset15 as op',
        '',
        '@onnxscript.script()',
        'def custom_model(X, Y):',
        '    return op.MatMul(X, Y)',
      ].join('\n'),
      language: 'python',
      theme: isDark ? 'vs-dark' : 'vs',
      automaticLayout: true,
      minimap: { enabled: false },
    });

    globalEvents.on('themeChanged', (theme: string) => {
      monaco.editor.setTheme(theme === 'dark' ? 'vs-dark' : 'vs');
    });

    this.editorInstance.onDidChangeModelContent(() => {
      this.handleContentChange();
    });
  }

  private handleContentChange(): void {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }

    this.debounceTimer = setTimeout(() => {
      const code = this.editorInstance.getValue();
      globalEvents.emit('onnxScriptChanged', code);
    }, 1000); // 1 second debounce
  }

  public highlightError(line: number, message: string): void {
    if (!this.editorInstance) return;

    const marker = {
      severity: monaco.MarkerSeverity.Error,
      startLineNumber: line,
      startColumn: 1,
      endLineNumber: line,
      endColumn: 100,
      message: message,
    };
    const model = this.editorInstance.getModel();
    monaco.editor.setModelMarkers(model, 'onnxscript', [marker]);
  }

  public clearErrors(): void {
    if (!this.editorInstance) return;
    const model = this.editorInstance.getModel();
    monaco.editor.setModelMarkers(model, 'onnxscript', []);
  }

  public getValue(): string {
    return this.editorInstance ? this.editorInstance.getValue() : '';
  }

  public setValue(val: string): void {
    if (this.editorInstance) {
      this.editorInstance.setValue(val);
    }
  }

  public setLanguage(lang: string): void {
    if (this.editorInstance) {
      monaco.editor.setModelLanguage(this.editorInstance.getModel(), lang);
    }
  }
}
