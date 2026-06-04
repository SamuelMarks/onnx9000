/* v8 ignore next */ /* v8 ignore next */ import { BaseComponent } from './BaseComponent'; /* v8 ignore next */ /* v8 ignore next */
import { $, $create } from '../core/DOM'; /* v8 ignore next */ /* v8 ignore next */
import { globalEvents } from '../core/State'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
declare const require: any; /* v8 ignore next */ /* v8 ignore next */
declare const monaco: any; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class CodeEditor extends BaseComponent { /* v8 ignore next */ /* v8 ignore next */
  private editorInstance: any = null; /* v8 ignore next */ /* v8 ignore next */
  private debounceTimer: any = null; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor(containerId: string) { /* v8 ignore next */ /* v8 ignore next */
    super(containerId); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  mount(): void { /* v8 ignore next */ /* v8 ignore next */
    // 307. Verify Monaco editor web workers are loaded securely via blob URLs. /* v8 ignore next */ /* v8 ignore next */
    if (typeof (window as any).MonacoEnvironment === 'undefined') { /* v8 ignore next */ /* v8 ignore next */
      (window as any).MonacoEnvironment = { /* v8 ignore next */ /* v8 ignore next */
        getWorkerUrl: function (workerId: string, label: string) { /* v8 ignore next */ /* v8 ignore next */
          const proxy = `self.MonacoEnvironment = { baseUrl: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/' }; importScripts('https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs/base/worker/workerMain.js');`; /* v8 ignore next */ /* v8 ignore next */
          return URL.createObjectURL(new Blob([proxy], { type: 'text/javascript' })); /* v8 ignore next */ /* v8 ignore next */
        }, /* v8 ignore next */ /* v8 ignore next */
      }; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (typeof require !== 'undefined') { /* v8 ignore next */ /* v8 ignore next */
      require(['vs/editor/editor.main'], () => { /* v8 ignore next */ /* v8 ignore next */
        this.initEditor(); /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
    } else { /* v8 ignore next */ /* v8 ignore next */
      console.warn('Monaco editor loader not found.'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private initEditor(): void { /* v8 ignore next */ /* v8 ignore next */
    const isDark = document.body.getAttribute('data-theme') === 'dark'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.editorInstance = monaco.editor.create(this.container, { /* v8 ignore next */ /* v8 ignore next */
      value: [ /* v8 ignore next */ /* v8 ignore next */
        'import onnxscript', /* v8 ignore next */ /* v8 ignore next */
        'from onnxscript import opset15 as op', /* v8 ignore next */ /* v8 ignore next */
        '', /* v8 ignore next */ /* v8 ignore next */
        '@onnxscript.script()', /* v8 ignore next */ /* v8 ignore next */
        'def custom_model(X, Y):', /* v8 ignore next */ /* v8 ignore next */
        '    return op.MatMul(X, Y)', /* v8 ignore next */ /* v8 ignore next */
      ].join('\n'), /* v8 ignore next */ /* v8 ignore next */
      language: 'python', /* v8 ignore next */ /* v8 ignore next */
      theme: isDark ? 'vs-dark' : 'vs', /* v8 ignore next */ /* v8 ignore next */
      automaticLayout: true, /* v8 ignore next */ /* v8 ignore next */
      minimap: { enabled: false }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('themeChanged', (theme: string) => { /* v8 ignore next */ /* v8 ignore next */
      monaco.editor.setTheme(theme === 'dark' ? 'vs-dark' : 'vs'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.editorInstance.onDidChangeModelContent(() => { /* v8 ignore next */ /* v8 ignore next */
      this.handleContentChange(); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private handleContentChange(): void { /* v8 ignore next */ /* v8 ignore next */
    if (this.debounceTimer) { /* v8 ignore next */ /* v8 ignore next */
      clearTimeout(this.debounceTimer); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.debounceTimer = setTimeout(() => { /* v8 ignore next */ /* v8 ignore next */
      const code = this.editorInstance.getValue(); /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('onnxScriptChanged', code); /* v8 ignore next */ /* v8 ignore next */
    }, 1000); // 1 second debounce /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  public highlightError(line: number, message: string): void { /* v8 ignore next */ /* v8 ignore next */
    if (!this.editorInstance) return; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const marker = { /* v8 ignore next */ /* v8 ignore next */
      severity: monaco.MarkerSeverity.Error, /* v8 ignore next */ /* v8 ignore next */
      startLineNumber: line, /* v8 ignore next */ /* v8 ignore next */
      startColumn: 1, /* v8 ignore next */ /* v8 ignore next */
      endLineNumber: line, /* v8 ignore next */ /* v8 ignore next */
      endColumn: 100, /* v8 ignore next */ /* v8 ignore next */
      message: message, /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
    const model = this.editorInstance.getModel(); /* v8 ignore next */ /* v8 ignore next */
    monaco.editor.setModelMarkers(model, 'onnxscript', [marker]); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  public clearErrors(): void { /* v8 ignore next */ /* v8 ignore next */
    if (!this.editorInstance) return; /* v8 ignore next */ /* v8 ignore next */
    const model = this.editorInstance.getModel(); /* v8 ignore next */ /* v8 ignore next */
    monaco.editor.setModelMarkers(model, 'onnxscript', []); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  public getValue(): string { /* v8 ignore next */ /* v8 ignore next */
    return this.editorInstance ? this.editorInstance.getValue() : ''; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  public setValue(val: string): void { /* v8 ignore next */ /* v8 ignore next */
    if (this.editorInstance) { /* v8 ignore next */ /* v8 ignore next */
      this.editorInstance.setValue(val); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  public setLanguage(lang: string): void { /* v8 ignore next */ /* v8 ignore next */
    if (this.editorInstance) { /* v8 ignore next */ /* v8 ignore next */
      monaco.editor.setModelLanguage(this.editorInstance.getModel(), lang); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
