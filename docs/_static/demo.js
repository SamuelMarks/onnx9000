/**
 * @file demo.js
 * @description Interactive demo logic for the ONNX9000 documentation, including tab switching and simulated model conversion with Monaco Editor.
 */

/**
 * Interactive demo application for ONNX9000.
 */
class ONNX9000Demo {
  /**
   * Initialize the demo application.
   * @param {Document} doc - The document object.
   * @param {Object} [options] - Optional configurations.
   * @param {number} [options.delay=800] - Mock conversion delay in milliseconds.
   */
  constructor(doc, options = {}) {
    this.doc = doc;
    this.delay = options.delay !== undefined ? options.delay : 800;
    this.inputMonacoEditor = null;
    this.outputMonacoEditor = null;

    /**
     * Mock inputs mapping
     * @type {Object.<string, {language: string, code: string}>}
     */
    this.mockInputs = {
      pytorch: {
        language: 'python',
        code: `import torch\nimport torch.nn as nn\n\nclass SimpleNet(nn.Module):\n    def __init__(self):\n        super(SimpleNet, self).__init__()\n        self.fc = nn.Linear(10, 2)\n        self.relu = nn.ReLU()\n\n    def forward(self, x):\n        return self.relu(self.fc(x))\n\nmodel = SimpleNet()`,
      },
      onnxscript: {
        language: 'python',
        code: `from onnxscript import script, opset15 as op\n\n@script()\ndef SimpleNet(x):\n    w = op.Constant(value_float=[[0.1, 0.2]])\n    b = op.Constant(value_float=[0.0, 0.0])\n    return op.Relu(op.MatMul(x, w) + b)`,
      },
    };

    /**
     * Mock outputs mapping
     * @type {Object.<string, {language: string, code: string}>}
     */
    this.mockOutputs = {
      onnx: {
        language: 'json',
        code: `{\n  "ir_version": 8,\n  "producer_name": "ONNX9000",\n  "graph": {\n    "name": "SimpleNet",\n    "node": [\n      {\n        "op_type": "MatMul",\n        "input": ["x", "fc.weight"],\n        "output": ["1"]\n      },\n      {\n        "op_type": "Add",\n        "input": ["1", "fc.bias"],\n        "output": ["2"]\n      },\n      {\n        "op_type": "Relu",\n        "input": ["2"],\n        "output": ["output"]\n      }\n    ]\n  }\n}`,
      },
      mlir: {
        language: 'plaintext',
        code: `module {\n  func.func @simple_net(%arg0: tensor<10xf32>) -> tensor<2xf32> {\n    %cst_w = "onnx.Constant"() {value = dense<[[0.1, 0.2]]> : tensor<10x2xf32>} : () -> tensor<10x2xf32>\n    %cst_b = "onnx.Constant"() {value = dense<[0.0, 0.0]> : tensor<2xf32>} : () -> tensor<2xf32>\n    %0 = "onnx.MatMul"(%arg0, %cst_w) : (tensor<10xf32>, tensor<10x2xf32>) -> tensor<2xf32>\n    %1 = "onnx.Add"(%0, %cst_b) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>\n    %2 = "onnx.Relu"(%1) : (tensor<2xf32>) -> tensor<2xf32>\n    return %2 : tensor<2xf32>\n  }\n}`,
      },
      c: {
        language: 'c',
        code: `#include <math.h>\n\nvoid simple_net(const float* x, float* out) {\n    // Static weights evaluated via ONNX9000 Codegen\n    const float w[10][2] = { ... };\n    const float b[2] = {0.0f, 0.0f};\n    \n    for (int i=0; i<2; ++i) {\n        float sum = b[i];\n        for (int j=0; j<10; ++j) {\n            sum += x[j] * w[j][i];\n        }\n        out[i] = sum > 0.0f ? sum : 0.0f; // ReLU\n    }\n}`,
      },
    };

    this.bindTabs();
    this.initMonaco();
  }

  /**
   * Helper to handle accessible keyboard interaction (Enter or Space) on tabs.
   * @param {KeyboardEvent} e - The keydown event.
   * @param {HTMLElement} tab - The tab element.
   */
  handleTabKeydown(e, tab) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      tab.click();
    }
  }

  /**
   * Generalized tab binding logic that supports ARIA roles.
   * @param {NodeListOf<Element>} tabs - The tab buttons.
   * @param {NodeListOf<Element>} panels - The target panels.
   */
  setupTabGroup(tabs, panels) {
    tabs.forEach((tab) => {
      tab.addEventListener('keydown', (e) => this.handleTabKeydown(e, tab));
      tab.addEventListener('click', () => {
        tabs.forEach((t) => {
          t.classList.remove('active');
          t.setAttribute('aria-selected', 'false');
          t.setAttribute('tabindex', '-1');
        });
        panels.forEach((p) => {
          p.classList.remove('active');
        });

        tab.classList.add('active');
        tab.setAttribute('aria-selected', 'true');
        tab.setAttribute('tabindex', '0');

        const targetId = tab.getAttribute('data-target');
        const targetPanel = this.doc.getElementById(targetId);
        if (targetPanel) {
          targetPanel.classList.add('active');
        }

        // Special handling for the main converter demo tab to resize Monaco
        if (targetId === 'converter-demo' && this.inputMonacoEditor && this.outputMonacoEditor) {
          this.inputMonacoEditor.layout();
          this.outputMonacoEditor.layout();
        }
      });
    });
  }

  /**
   * Bind event listeners for tab switching.
   */
  bindTabs() {
    // Top-level demo tabs
    const demoTabs = this.doc.querySelectorAll('.demo-tab');
    const demoPanels = this.doc.querySelectorAll('.demo-container > .demo-panel');
    this.setupTabGroup(demoTabs, demoPanels);

    // Bottom pane nested tabs
    const bottomTabs = this.doc.querySelectorAll('.bottom-tab');
    const bottomPanels = this.doc.querySelectorAll('.bottom-panel');
    if (bottomTabs.length > 0 && bottomPanels.length > 0) {
      this.setupTabGroup(bottomTabs, bottomPanels);
    }
  }

  /**
   * Determine current active theme based on document body data attributes.
   * Defaults to 'vs-light' if dark mode is not explicitly enabled.
   * @returns {string} Monaco theme identifier.
   */
  getTheme() {
    return this.doc.body.dataset.theme === 'dark' ? 'vs-dark' : 'vs-light';
  }

  /**
   * Add text to the conversion log panel.
   * @param {string} message - Message to log.
   */
  appendLog(message) {
    const logPanel = this.doc.getElementById('conversion-log');
    if (logPanel) {
      const timestamp = new Date().toISOString().split('T')[1].slice(0, -1);
      logPanel.textContent += `\n[${timestamp}] ${message}`;
      logPanel.scrollTop = logPanel.scrollHeight;
    }
  }

  /**
   * Initialize Monaco editor instances and bind editor events.
   */
  initMonaco() {
    const inputLang = this.doc.getElementById('input-lang');
    const outputLang = this.doc.getElementById('output-lang');
    const convertBtn = this.doc.getElementById('convert-btn');

    if (typeof globalThis.require !== 'undefined' && inputLang && outputLang && convertBtn) {
      globalThis.require(['vs/editor/editor.main'], () => {
        let currentTheme = this.getTheme();

        // ARIA labels are partially supported natively by Monaco via ariaLabel constructor option
        this.inputMonacoEditor = globalThis.monaco.editor.create(
          this.doc.getElementById('input-editor'),
          {
            value: this.mockInputs[inputLang.value] ? this.mockInputs[inputLang.value].code : '',
            language: this.mockInputs[inputLang.value]
              ? this.mockInputs[inputLang.value].language
              : 'plaintext',
            theme: currentTheme,
            automaticLayout: true,
            minimap: { enabled: false },
            fontSize: 13,
            scrollBeyondLastLine: false,
            ariaLabel: 'Source code editor input',
          },
        );

        this.outputMonacoEditor = globalThis.monaco.editor.create(
          this.doc.getElementById('output-editor'),
          {
            value: '// Click convert to compile to target format',
            language: 'plaintext',
            theme: currentTheme,
            automaticLayout: true,
            readOnly: true,
            minimap: { enabled: false },
            fontSize: 13,
            scrollBeyondLastLine: false,
            ariaLabel: 'Converted source code output (Read only)',
          },
        );

        const observer = new MutationObserver((mutations) => {
          mutations.forEach((mutation) => {
            if (mutation.attributeName === 'data-theme') {
              globalThis.monaco.editor.setTheme(this.getTheme());
            }
          });
        });
        observer.observe(this.doc.body, { attributes: true });

        inputLang.addEventListener('change', (e) => {
          const val = this.mockInputs[e.target.value];
          if (val) {
            globalThis.monaco.editor.setModelLanguage(
              this.inputMonacoEditor.getModel(),
              val.language,
            );
            this.inputMonacoEditor.setValue(val.code);
            this.appendLog(`Switched input language to: ${e.target.value}`);
          }
          this.outputMonacoEditor.setValue('// Source changed. Click convert to compile.');
        });

        outputLang.addEventListener('change', (e) => {
          this.outputMonacoEditor.setValue('// Target changed. Click convert to compile.');
          this.appendLog(`Switched target language to: ${e.target.value}`);
        });

        convertBtn.addEventListener('click', () => {
          this.performConversion(convertBtn, outputLang.value);
        });
      });
    }
  }

  /**
   * Simulate a code conversion process.
   * @param {HTMLButtonElement} convertBtn - The trigger button to update state.
   * @param {string} targetValue - The target output language key.
   */
  performConversion(convertBtn, targetValue) {
    convertBtn.disabled = true;
    convertBtn.innerText = 'Compiling...';

    this.appendLog('Starting compilation request...');
    this.appendLog(`Resolving AST for target backend: ${targetValue}`);

    if (this.outputMonacoEditor) {
      this.outputMonacoEditor.setValue('// Analyzing IR tree and compiling...');
    }

    setTimeout(() => {
      const target = this.mockOutputs[targetValue];
      if (this.outputMonacoEditor) {
        if (target) {
          globalThis.monaco.editor.setModelLanguage(
            this.outputMonacoEditor.getModel(),
            target.language,
          );
          this.outputMonacoEditor.setValue(target.code);
          this.appendLog('Compilation successful. Code generated.');
        } else {
          this.outputMonacoEditor.setValue('// Target not fully implemented in mock.');
          this.appendLog('Compilation warning: Target not fully implemented in mock.');
        }
      }

      convertBtn.disabled = false;
      convertBtn.innerHTML = 'Convert &rarr;';
    }, this.delay);
  }
}

// Export for testability (Node/JSDOM environment) or auto-init in browser
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ONNX9000Demo;
} else if (typeof window !== 'undefined') {
  window.addEventListener('DOMContentLoaded', () => {
    window.onnxDemo = new ONNX9000Demo(document);
  });
}
