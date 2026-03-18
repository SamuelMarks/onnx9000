// @vitest-environment jsdom

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import ONNX9000Demo from '../_static/demo.js';

describe('ONNX9000Demo', () => {
  let doc;
  let demo;

  beforeEach(() => {
    // Setup basic DOM
    doc = document.implementation.createHTMLDocument();
    doc.body.dataset.theme = 'light';
    doc.body.innerHTML = `
      <div class="demo-container">
        <div class="demo-tabs">
          <button class="demo-tab active" data-target="converter-demo">Model Converter</button>
          <button class="demo-tab" data-target="netron-demo">Netron UI</button>
        </div>

        <div id="converter-demo" class="demo-panel active">
          <select id="input-lang">
            <option value="pytorch">PyTorch</option>
            <option value="onnxscript">ONNX Script</option>
          </select>
          <button id="convert-btn">Convert</button>
          <select id="output-lang">
            <option value="onnx">ONNX</option>
            <option value="c">C</option>
          </select>
          <div id="input-editor"></div>
          <div id="output-editor"></div>
          
          <div class="bottom-tabs">
            <button class="bottom-tab active" data-target="panel-visualize">Visualize</button>
            <button class="bottom-tab" data-target="panel-log">Log</button>
          </div>
          <div id="panel-visualize" class="bottom-panel active"></div>
          <div id="panel-log" class="bottom-panel">
            <div id="conversion-log"></div>
          </div>
        </div>

        <div id="netron-demo" class="demo-panel"></div>
      </div>
    `;

    // Mock global requires and monaco
    globalThis.require = vi.fn((deps, callback) => callback());

    // Create mock monaco models and editors
    const mockModel = {
      setValue: vi.fn(),
      getValue: vi.fn(),
    };

    const mockEditor = {
      getModel: vi.fn(() => mockModel),
      setValue: vi.fn(),
      layout: vi.fn(),
    };

    globalThis.monaco = {
      editor: {
        create: vi.fn(() => ({
          ...mockEditor,
          // create separate mocks for each instance if needed,
          // but for basic coverage returning a generic mock is fine
          setValue: vi.fn(),
          layout: vi.fn(),
          getModel: vi.fn(() => mockModel),
        })),
        setTheme: vi.fn(),
        setModelLanguage: vi.fn(),
      },
    };

    demo = new ONNX9000Demo(doc, { delay: 10 }); // Small delay for fast testing
  });

  afterEach(() => {
    delete globalThis.require;
    delete globalThis.monaco;
    vi.clearAllTimers();
  });

  it('initializes the demo correctly', () => {
    expect(demo).toBeDefined();
    expect(globalThis.require).toHaveBeenCalled();
    expect(globalThis.monaco.editor.create).toHaveBeenCalledTimes(2);
  });

  it('determines the correct theme based on document body', () => {
    expect(demo.getTheme()).toBe('vs-light');
    doc.body.dataset.theme = 'dark';
    expect(demo.getTheme()).toBe('vs-dark');
  });

  it('switches main tabs correctly by click', () => {
    const netronTab = doc.querySelector('[data-target="netron-demo"]');
    const converterTab = doc.querySelector('[data-target="converter-demo"]');

    // Simulate click on Netron tab
    netronTab.click();
    expect(netronTab.classList.contains('active')).toBe(true);
    expect(doc.getElementById('netron-demo').classList.contains('active')).toBe(true);
    expect(converterTab.classList.contains('active')).toBe(false);
    expect(doc.getElementById('converter-demo').classList.contains('active')).toBe(false);

    // Click back to converter demo, should trigger layout on Monaco
    const layoutSpy = vi.spyOn(demo.inputMonacoEditor, 'layout');
    converterTab.click();
    expect(converterTab.classList.contains('active')).toBe(true);
    expect(layoutSpy).toHaveBeenCalled();
  });

  it('switches tabs correctly by keyboard (Enter/Space)', () => {
    const netronTab = doc.querySelector('[data-target="netron-demo"]');

    // Test Enter key
    let enterEvent = new KeyboardEvent('keydown', { key: 'Enter' });
    let preventDefaultSpy = vi.spyOn(enterEvent, 'preventDefault');
    netronTab.dispatchEvent(enterEvent);

    expect(preventDefaultSpy).toHaveBeenCalled();
    expect(doc.getElementById('netron-demo').classList.contains('active')).toBe(true);

    const converterTab = doc.querySelector('[data-target="converter-demo"]');

    // Test Space key
    let spaceEvent = new KeyboardEvent('keydown', { key: ' ' });
    preventDefaultSpy = vi.spyOn(spaceEvent, 'preventDefault');
    converterTab.dispatchEvent(spaceEvent);

    expect(preventDefaultSpy).toHaveBeenCalled();
    expect(doc.getElementById('converter-demo').classList.contains('active')).toBe(true);
  });

  it('switches bottom tabs correctly by click', () => {
    const logTab = doc.querySelector('.bottom-tab[data-target="panel-log"]');
    const vizTab = doc.querySelector('.bottom-tab[data-target="panel-visualize"]');

    logTab.click();
    expect(logTab.classList.contains('active')).toBe(true);
    expect(doc.getElementById('panel-log').classList.contains('active')).toBe(true);
    expect(vizTab.classList.contains('active')).toBe(false);
    expect(doc.getElementById('panel-visualize').classList.contains('active')).toBe(false);
  });

  it('handles input language change and logs', () => {
    const inputLang = doc.getElementById('input-lang');
    const setValueSpy = vi.spyOn(demo.inputMonacoEditor, 'setValue');

    inputLang.value = 'onnxscript';
    inputLang.dispatchEvent(new Event('change'));

    expect(globalThis.monaco.editor.setModelLanguage).toHaveBeenCalled();
    expect(setValueSpy).toHaveBeenCalled();
    expect(doc.getElementById('conversion-log').textContent).toContain(
      'Switched input language to: onnxscript',
    );
  });

  it('handles output language change and logs', () => {
    const outputLang = doc.getElementById('output-lang');
    const setValueSpy = vi.spyOn(demo.outputMonacoEditor, 'setValue');

    outputLang.value = 'c';
    outputLang.dispatchEvent(new Event('change'));

    expect(setValueSpy).toHaveBeenCalledWith('// Target changed. Click convert to compile.');
    expect(doc.getElementById('conversion-log').textContent).toContain(
      'Switched target language to: c',
    );
  });

  it('performs conversion simulation correctly and updates log', async () => {
    vi.useFakeTimers();

    const convertBtn = doc.getElementById('convert-btn');
    const setValueSpy = vi.spyOn(demo.outputMonacoEditor, 'setValue');

    convertBtn.click();

    // Assert immediate state
    expect(convertBtn.disabled).toBe(true);
    expect(setValueSpy).toHaveBeenCalledWith('// Analyzing IR tree and compiling...');
    expect(doc.getElementById('conversion-log').textContent).toContain(
      'Starting compilation request...',
    );

    // Fast-forward timeout
    vi.advanceTimersByTime(15);

    // Assert resolved state
    expect(convertBtn.disabled).toBe(false);
    expect(setValueSpy).toHaveBeenCalledWith(demo.mockOutputs['onnx'].code);
    expect(doc.getElementById('conversion-log').textContent).toContain(
      'Compilation successful. Code generated.',
    );

    vi.useRealTimers();
  });

  it('handles conversion missing mock correctly and logs warning', async () => {
    vi.useFakeTimers();

    const convertBtn = doc.getElementById('convert-btn');
    const outputLang = doc.getElementById('output-lang');
    const setValueSpy = vi.spyOn(demo.outputMonacoEditor, 'setValue');

    // Make target an unsupported one
    outputLang.innerHTML += '<option value="unsupported">Unsupported</option>';
    outputLang.value = 'unsupported';

    convertBtn.click();

    // Fast-forward timeout
    vi.advanceTimersByTime(15);

    // Assert resolved state
    expect(setValueSpy).toHaveBeenCalledWith('// Target not fully implemented in mock.');
    expect(doc.getElementById('conversion-log').textContent).toContain(
      'Compilation warning: Target not fully implemented in mock.',
    );

    vi.useRealTimers();
  });

  it('observes theme changes via MutationObserver', async () => {
    // Modify body attribute to trigger observer
    doc.body.setAttribute('data-theme', 'dark');

    // Wait a tick for mutation observer
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(globalThis.monaco.editor.setTheme).toHaveBeenCalledWith('vs-dark');
  });

  it('auto-initializes in browser environment', async () => {
    const fs = require('fs');
    const path = require('path');
    const code = fs.readFileSync(path.join(__dirname, '../_static/demo.js'), 'utf-8');

    const mockWindow = {
      addEventListener: vi.fn((event, cb) => {
        if (event === 'DOMContentLoaded') cb();
      }),
    };

    const evalScope = new Function('window', 'document', 'module', 'globalThis', code);
    evalScope(mockWindow, doc, undefined, globalThis);

    expect(mockWindow.addEventListener).toHaveBeenCalledWith(
      'DOMContentLoaded',
      expect.any(Function),
    );
    expect(mockWindow.onnxDemo).toBeDefined();
  });
});
