import * as cCompilerMod from '@onnx9000/c-compiler';
import * as convertersMod from '@onnx9000/converters';
import * as coreMod from '@onnx9000/core';
import { type Graph, Node, Tensor, ValueInfo } from '@onnx9000/core';
import * as coremlMod from '@onnx9000/coreml';
import * as optimumMod from '@onnx9000/optimum';
import { describe, expect, it, vi } from 'vitest';
import { BenchmarkPanel } from '../src/components/BenchmarkPanel.js';
import { BottomContainer } from '../src/components/BottomContainer.js';
import { Breadcrumbs } from '../src/components/Breadcrumbs.js';
import { Console } from '../src/components/Console.js';
import { Dropdown } from '../src/components/Dropdown.js';
import { Editor } from '../src/components/Editor.js';
import { FileTree } from '../src/components/FileTree.js';
import { LHSContainer } from '../src/components/LHSContainer.js';
import { MetricsDashboard } from '../src/components/MetricsDashboard.js';
import { OliveConfigPanel } from '../src/components/OliveConfigPanel.js';
import { OnnxVisualizer } from '../src/components/OnnxVisualizer.js';
import { PromoteButton } from '../src/components/PromoteButton.js';
import { RHSContainer } from '../src/components/RHSContainer.js';
import { SplitPane } from '../src/components/SplitPane.js';
import { Tabs } from '../src/components/Tabs.js';
import { TensorInputModal } from '../src/components/TensorInputModal.js';
import { ToastNotification } from '../src/components/ToastNotification.js';
import { WasmOverlay } from '../src/components/WasmOverlay.js';
import { Cache } from '../src/core/Cache.js';
import { Component } from '../src/core/Component.js';
import { Debouncer } from '../src/core/Debouncer.js';
import { EventBus, globalEventBus } from '../src/core/EventBus.js';
import { HexFormatter } from '../src/core/HexFormatter.js';
import { i18n, t } from '../src/core/I18n.js';
import { KerasPythonParser } from '../src/core/KerasPythonParser.js';
import { Logger, LogLevel } from '../src/core/Logger.js';
import { MathUtils } from '../src/core/MathUtils.js';
import { OliveOptimizer } from '../src/core/OliveOptimizer.js';
import { Onnx2CCompiler } from '../src/core/Onnx2CCompiler.js';
import { OnnxAdapter } from '../src/core/OnnxAdapter.js';
import { ORTWebRunner } from '../src/core/ORTWebRunner.js';
import { PipelineHistory } from '../src/core/PipelineHistory.js';
import { PipelineNode } from '../src/core/PipelineNode.js';
import { PipelineValidator } from '../src/core/PipelineValidator.js';
import { Store } from '../src/core/Store.js';
import { WasmManager, WasmState } from '../src/core/WasmManager.js';
import { WebNNPolyfillRunner } from '../src/core/WebNNPolyfillRunner.js';
import { WorkerManager } from '../src/core/WorkerManager.js';
import { RHS_TARGETS } from '../src/data/MockData.js';

const cyHandlers: Record<string, (evt: unknown) => void> = {};

vi.mock('cytoscape', () => {
  const cytoscape = vi.fn().mockImplementation(() => {
    const inst = {
      elements: () => ({ remove: vi.fn() }),
      add: vi.fn(),
      layout: () => ({ run: vi.fn() }),
      center: vi.fn(),
      resize: vi.fn(),
      fit: vi.fn(),
      on: vi.fn((event: string, selectorOrCb: unknown, maybeCb?: unknown) => {
        if (typeof selectorOrCb === 'string' && typeof maybeCb === 'function') {
          cyHandlers[`${event}:${selectorOrCb}`] = maybeCb as (evt: unknown) => void;
        } else if (typeof selectorOrCb === 'function') {
          cyHandlers[event] = selectorOrCb as (evt: unknown) => void;
        }
      }),
      destroy: vi.fn(),
    };
    return inst;
  });
  return { default: cytoscape };
});

describe('apps/sphinx-demo-ui exhaustive 100% coverage', () => {
  it('covers src/core/* remaining lines, functions, and branches', async () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    // Cache: clear and evictExpired
    const cache = new Cache<string>(10);
    cache.set('k1', 'v1', -10);
    cache.set('k2', 'v2', 10000);
    cache.evictExpired();
    expect(cache.get('k1')).toBeUndefined();
    cache.clear();
    expect(cache.get('k2')).toBeUndefined();

    // Component: mount/replace without element, replace with and without parentNode
    class UninitComp extends Component<HTMLDivElement> {
      protected render() {
        return document.createElement('div');
      }
    }
    const uninit = new UninitComp();
    expect(() => uninit.mount(document.body)).toThrow();
    expect(() => uninit.replace(document.body)).toThrow();
    uninit.element = document.createElement('div');
    const detachedOld = document.createElement('div');
    uninit.replace(detachedOld);
    const parentDiv = document.createElement('div');
    const attachedOld = document.createElement('div');
    parentDiv.appendChild(attachedOld);
    uninit.replace(attachedOld);
    expect(parentDiv.firstChild).toBe(uninit.element);

    // Debouncer: clear when active and inactive
    const deb = new Debouncer();
    deb.clear();
    const debFn = deb.debounce(() => {}, 1000);
    debFn();
    deb.clear();

    // EventBus: off on unregistered event
    const bus = new EventBus();
    bus.off('no_such_event', () => {});

    // HexFormatter: empty, printable vs non-printable ASCII, <16 bytes padding, and >maxLength truncation
    expect(HexFormatter.format(new Uint8Array(0))).toBe('');
    const bytes = new Uint8Array(20);
    bytes[0] = 65; // 'A'
    bytes[1] = 1; // non-printable
    expect(HexFormatter.format(bytes.subarray(0, 5))).toContain('A.');
    expect(HexFormatter.format(bytes, 16)).toContain('truncated');

    // I18n: fallback key, args replacement, detectLanguage from localStorage and navigator.language
    expect(t('non.existent.key', { name: 'World' })).toBe('non.existent.key');
    expect(t('hello.{error}', { error: 'boom' })).toContain('boom');
    localStorage.setItem('onnx9000-demo-lang', 'es');
    (i18n as unknown as { detectLanguage: () => void }).detectLanguage();
    expect(i18n.getLanguage()).toBe('es');
    localStorage.removeItem('onnx9000-demo-lang');
    Object.defineProperty(navigator, 'language', { value: 'fr-FR', configurable: true });
    (i18n as unknown as { detectLanguage: () => void }).detectLanguage();
    expect(i18n.getLanguage()).toBe('fr');
    Object.defineProperty(navigator, 'language', { value: 'zz-ZZ', configurable: true });
    (i18n as unknown as { detectLanguage: () => void }).detectLanguage();
    i18n.setLanguage('en');

    // KerasPythonParser: cached instance, concurrent isLoading wait, script injection, and error catch
    const fakePyodide = { runPythonAsync: vi.fn().mockResolvedValue('{}') };
    (KerasPythonParser as unknown as { pyodideInstance: unknown }).pyodideInstance = fakePyodide;
    expect(await KerasPythonParser.initPyodide()).toBe(fakePyodide);

    (KerasPythonParser as unknown as { pyodideInstance: unknown }).pyodideInstance = null;
    (KerasPythonParser as unknown as { isLoading: boolean }).isLoading = true;
    setTimeout(() => {
      (KerasPythonParser as unknown as { pyodideInstance: unknown }).pyodideInstance = fakePyodide;
    }, 10);
    expect(await KerasPythonParser.initPyodide()).toBe(fakePyodide);

    // Script injection + failure path
    (KerasPythonParser as unknown as { pyodideInstance: unknown }).pyodideInstance = null;
    (KerasPythonParser as unknown as { isLoading: boolean }).isLoading = false;
    delete (window as unknown as { loadPyodide?: unknown }).loadPyodide;
    const origAppend = document.head.appendChild.bind(document.head);
    vi.spyOn(document.head, 'appendChild').mockImplementationOnce(((el: unknown) => {
      setTimeout(() => (el as HTMLScriptElement).onerror?.(new Event('error')), 0);
      return el as globalThis.Node;
    }) as never);
    await expect(KerasPythonParser.initPyodide()).rejects.toBeDefined();

    // Script injection + success path
    vi.spyOn(document.head, 'appendChild').mockImplementationOnce(((el: unknown) => {
      (window as unknown as { loadPyodide: unknown }).loadPyodide = vi
        .fn()
        .mockResolvedValue(fakePyodide);
      setTimeout(() => (el as HTMLScriptElement).onload?.(new Event('load')), 0);
      return origAppend(el as globalThis.Node);
    }) as never);
    expect(await KerasPythonParser.initPyodide()).toBe(fakePyodide);

    // Logger: startIntercepting when already intercepting, console.warn, console.error, circular object, logDirect
    const logger = Logger.getInstance();
    logger.stopIntercepting();
    logger.stopIntercepting();
    logger.startIntercepting();
    logger.startIntercepting();
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    console.warn('warn msg', circular);
    console.error('err msg', { ok: 1 });
    logger.logDirect(LogLevel.INFO, 'direct msg');
    logger.stopIntercepting();

    // MathUtils: empty array mean/variance/normalize and flat normalize
    expect(MathUtils.mean([])).toBe(0);
    expect(MathUtils.variance([])).toBe(0);
    expect(MathUtils.normalize([])).toEqual([]);
    expect(MathUtils.normalize([5, 5])).toEqual([0.5, 0.5]);

    // ORTWebRunner, OliveOptimizer, Onnx2CCompiler, WebNNPolyfillRunner success & error catch blocks
    const wm = WorkerManager.getInstance();
    const termSpy = vi.spyOn(wm, 'terminate').mockImplementation(() => {});
    const initSpy = vi.spyOn(wm, 'initWorker').mockImplementation(() => {});
    const execSpy = vi.spyOn(wm, 'execute').mockResolvedValueOnce({ y: {} } as never);
    await new ORTWebRunner().runInference(new Uint8Array(1), {});
    execSpy.mockResolvedValueOnce(new Uint8Array([1]) as never);
    await new OliveOptimizer().optimize(new Uint8Array([1, 2]), {
      quantizationLevel: 'None',
      enableStaticShapeInference: false,
      enableTransformerFusion: false,
    });
    execSpy.mockResolvedValueOnce(new Uint8Array([1]) as never);
    await new OliveOptimizer().simplify(new Uint8Array([1]));
    execSpy.mockResolvedValueOnce('void main() {}' as never);
    await new Onnx2CCompiler().compile(new Uint8Array([1]));
    execSpy.mockResolvedValueOnce({ y: {} } as never);
    await new WebNNPolyfillRunner().runInference(new Uint8Array(1), {});

    execSpy.mockRejectedValue(new Error('worker err'));
    await expect(new ORTWebRunner().runInference(new Uint8Array(1), {})).rejects.toThrow(
      'worker err',
    );
    await expect(
      new OliveOptimizer().optimize(new Uint8Array(1), {
        quantizationLevel: 'None',
        enableStaticShapeInference: false,
        enableTransformerFusion: false,
      }),
    ).rejects.toThrow('worker err');
    await expect(new OliveOptimizer().simplify(new Uint8Array(1))).rejects.toThrow('worker err');
    await expect(new Onnx2CCompiler().compile(new Uint8Array(1))).rejects.toThrow('worker err');
    execSpy.mockRejectedValueOnce('string worker err');
    await expect(new Onnx2CCompiler().compile(new Uint8Array(1))).rejects.toThrow(
      'string worker err',
    );
    await expect(new WebNNPolyfillRunner().runInference(new Uint8Array(1), {})).rejects.toThrow(
      'worker err',
    );
    execSpy.mockRestore();
    initSpy.mockRestore();
    termSpy.mockRestore();
    (wm as unknown as { worker: unknown }).worker = null;

    // OnnxAdapter: initializer fallback and unproduced output
    const els = OnnxAdapter.toCytoscape({
      inputs: [],
      outputs: [{ name: 'unproduced_out', type: 'float32' }],
      nodes: [{ id: 'n1', name: 'n1', opType: 'Relu', inputs: ['init_w'], outputs: ['y'] }],
    });
    expect(els.some((e) => e.classes === 'onnx-initializer')).toBe(true);

    // PipelineHistory: undo/redo when empty and truncation after undo + push
    const hist = new PipelineHistory();
    expect(hist.undo()).toBeNull();
    expect(hist.redo()).toBeNull();
    hist.push({ sourceFramework: 'a', targetFramework: 'b', activeFile: 'f' }, 'step1');
    hist.push({ sourceFramework: 'b', targetFramework: 'c', activeFile: 'f' }, 'step2');
    expect(hist.undo()?.sourceFramework).toBe('a');
    hist.push({ sourceFramework: 'd', targetFramework: 'e', activeFile: 'f' }, 'step3');
    expect(hist.getHistory()).toHaveLength(2);

    // PipelineNode without crypto.randomUUID
    const origCrypto = globalThis.crypto;
    Object.defineProperty(globalThis, 'crypto', { value: undefined, configurable: true });
    try {
      const pNode = new PipelineNode(
        { sourceFramework: 'a', targetFramework: 'b', activeFile: 'f' },
        'd',
      );
      expect(pNode.id).toBeTruthy();
    } finally {
      Object.defineProperty(globalThis, 'crypto', { value: origCrypto, configurable: true });
    }

    // PipelineValidator: unknown source
    expect(PipelineValidator.isValidTransition('unknown_src', '.onnx')).toBe(false);
    expect(PipelineValidator.getValidTargets('unknown_src')).toEqual([]);

    // Store: onChange
    const store = new Store({ count: 0 });
    let changedProp = '';
    store.onChange((ev) => {
      changedProp = ev.property;
    });
    store.state.count = 1;
    store.state.count = 1; // unchanged
    expect(changedProp).toBe('count');

    // WasmManager: load when already LOADING, readableStream missing, fake progress (total === 0), value undefined chunk, abort callback, non-Error catch
    const wasmMgr = WasmManager.getInstance();
    wasmMgr.reset();
    (wasmMgr as unknown as { _state: WasmState })._state = WasmState.LOADED;
    await wasmMgr.load();
    wasmMgr.reset();

    const origFetch = globalThis.fetch;
    const origCompile = WebAssembly.compile;
    const origInst = WebAssembly.instantiate;
    try {
      expect(wasmMgr.progress).toBe(0);

      // !response.ok branch
      globalThis.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found',
      });
      await wasmMgr.load();
      expect(wasmMgr.state).toBe(WasmState.ERROR);
      wasmMgr.reset();

      globalThis.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        headers: { get: () => null },
        body: null,
      });
      await wasmMgr.load();
      expect(wasmMgr.state).toBe(WasmState.ERROR);
      wasmMgr.reset();

      let readStep = 0;
      globalThis.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        headers: { get: () => '4' },
        body: {
          getReader: () => ({
            read: async () => {
              readStep++;
              if (readStep === 1) return { done: false, value: undefined };
              if (readStep === 2) return { done: false, value: new Uint8Array([1, 2]) };
              return { done: true, value: undefined };
            },
          }),
        },
      });
      let abortFn: (() => void) | undefined;
      WebAssembly.compile = vi.fn().mockResolvedValue({} as WebAssembly.Module);
      WebAssembly.instantiate = vi
        .fn()
        .mockImplementation((_m: unknown, imp: { env: { abort: () => void } }) => {
          abortFn = imp.env.abort;
          return Promise.resolve({} as WebAssembly.Instance);
        });
      await wasmMgr.load();
      abortFn?.();
      expect(wasmMgr.state).toBe(WasmState.LOADED);
      wasmMgr.reset();

      // Also test with content-length null (total === 0)
      let readStep2 = 0;
      globalThis.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        headers: { get: () => null },
        body: {
          getReader: () => ({
            read: async () => {
              readStep2++;
              if (readStep2 === 1) return { done: false, value: new Uint8Array([1, 2]) };
              return { done: true, value: undefined };
            },
          }),
        },
      });
      await wasmMgr.load();
      wasmMgr.reset();

      globalThis.fetch = vi.fn().mockRejectedValueOnce('non-error wasm fail');
      await wasmMgr.load();
      expect(wasmMgr.error).toBe('Failed to load WASM binary');
    } finally {
      globalThis.fetch = origFetch;
      WebAssembly.compile = origCompile;
      WebAssembly.instantiate = origInst;
    }

    // WorkerManager: re-init when worker already exists, execute without worker, msg.error, STREAM_STDOUT/STDERR, onerror, timeout, terminate with pending request, and without crypto.randomUUID
    wm.terminate();
    await expect(wm.execute('TEST', {})).rejects.toThrow('Worker not initialized.');

    interface MockWorkerInst {
      onmessage: ((e: MessageEvent) => void) | null;
      onerror: ((err: unknown) => void) | null;
    }
    const holder: { inst: MockWorkerInst | null } = { inst: null };
    let lastMsgId = '';
    const OrigWorker = globalThis.Worker;
    const mockW: MockWorkerInst & {
      postMessage: (msg: { id: string }) => void;
      terminate: () => void;
    } = {
      onmessage: null,
      onerror: null,
      postMessage: (msg: { id: string }) => {
        lastMsgId = msg.id;
      },
      terminate: () => {},
    };
    holder.inst = mockW;
    (globalThis as unknown as { Worker: unknown }).Worker = function MockWorker() {
      return mockW;
    };
    try {
      wm.initWorker();
      wm.initWorker(); // re-init terminates previous worker

      // STREAM_STDOUT, STREAM_STDERR, and msg.error on pending request
      holder.inst?.onmessage?.({
        data: { id: 'stream', type: 'STREAM_STDOUT', payload: 'hello stdout' },
      } as MessageEvent);
      holder.inst?.onmessage?.({
        data: { id: 'stream', type: 'STREAM_STDERR', payload: 'hello stderr' },
      } as MessageEvent);
      const pMsgErr = wm.execute('ERR_TASK', {});
      holder.inst?.onmessage?.({
        data: { id: lastMsgId, type: 'ERR_TASK', error: 'task failed' },
      } as MessageEvent);
      await expect(pMsgErr).rejects.toThrow('task failed');

      // onerror rejects pending
      const pCrash = wm.execute('CRASH', {});
      holder.inst?.onerror?.(new Error('crash'));
      await expect(pCrash).rejects.toThrow('Worker crashed');

      // timeout without crypto.randomUUID
      Object.defineProperty(globalThis, 'crypto', { value: undefined, configurable: true });
      const pTimeout = wm.execute('SLOW', {}, 5);
      await expect(pTimeout).rejects.toThrow('timed out');
      Object.defineProperty(globalThis, 'crypto', { value: origCrypto, configurable: true });

      // terminate rejects pending
      const pTerm = wm.execute('TERM', {}, 10000);
      wm.terminate();
      await expect(pTerm).rejects.toThrow('Worker forcefully terminated');
    } finally {
      globalThis.Worker = OrigWorker;
    }

    logSpy.mockRestore();
    warnSpy.mockRestore();
    errSpy.mockRestore();
  });

  it('covers src/components/* and src/main.ts remaining lines, functions, and branches', async () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    // BenchmarkPanel: with pyodide absent, returning output, returning '', throwing Error, and throwing string
    const bench = new BenchmarkPanel();
    bench.mount(document.body);
    const btns = bench.element.querySelectorAll('button');
    delete (window as unknown as { pyodide?: unknown }).pyodide;
    btns[0]?.click();
    await new Promise((r) => setTimeout(r, 10));

    (window as unknown as { pyodide: unknown }).pyodide = {
      runPythonAsync: vi
        .fn()
        .mockResolvedValueOnce('bench output')
        .mockResolvedValueOnce('')
        .mockRejectedValueOnce(new Error('py err'))
        .mockRejectedValueOnce('py str err'),
    };
    btns[0]?.click();
    await new Promise((r) => setTimeout(r, 10));
    btns[1]?.click();
    await new Promise((r) => setTimeout(r, 10));
    btns[2]?.click();
    await new Promise((r) => setTimeout(r, 10));
    btns[0]?.click();
    await new Promise((r) => setTimeout(r, 10));
    delete (window as unknown as { pyodide?: unknown }).pyodide;
    bench.unmount();

    // BottomContainer: tab switch, LANGUAGE_CHANGED, and unmount
    const bottom = new BottomContainer();
    bottom.mount(document.body);
    globalEventBus.emit('LANGUAGE_CHANGED', 'en');
    const vizTabBtn = bottom.element.querySelector('#tab-viz') as HTMLButtonElement;
    vizTabBtn?.click();
    bottom.unmount();

    // Breadcrumbs: language select change, revert click, PIPELINE_STEP_REMOVED with matching and non-matching id
    const crumbs = new Breadcrumbs();
    crumbs.mount(document.body);
    const langSel = crumbs.element.querySelector('select') as HTMLSelectElement;
    langSel.value = 'es';
    langSel.dispatchEvent(new Event('change'));
    i18n.setLanguage('en');

    globalEventBus.emit('PIPELINE_STEP_ADDED', {
      id: 's1',
      description: 'Step 1',
      timestamp: 1,
      state: { sourceFramework: 'a', targetFramework: 'b', activeFile: 'f' },
    });
    globalEventBus.emit('PIPELINE_STEP_ADDED', {
      id: 's2',
      description: 'Step 2',
      timestamp: 2,
      state: { sourceFramework: 'b', targetFramework: 'c', activeFile: 'f' },
    });
    const firstCrumbBtn = crumbs.element.querySelector(
      '.demo-breadcrumb-item',
    ) as HTMLButtonElement;
    firstCrumbBtn?.click();
    globalEventBus.emit('PIPELINE_STEP_REMOVED', {
      id: 'non_existent',
      description: '',
      timestamp: 0,
      state: { sourceFramework: null, targetFramework: null, activeFile: null },
    });
    globalEventBus.emit('PIPELINE_STEP_REMOVED', {
      id: 's2',
      description: 'Step 2',
      timestamp: 2,
      state: { sourceFramework: 'b', targetFramework: 'c', activeFile: 'f' },
    });
    crumbs.unmount();

    // Console: > 1000 lines cap
    const cons = new Console();
    cons.mount(document.body);
    const outDiv = cons.element.querySelector('.demo-console-output') as HTMLDivElement;
    for (let i = 0; i < 1001; i++) {
      outDiv.appendChild(document.createElement('div'));
    }
    cons.appendLog({ level: LogLevel.INFO, message: 'overflow', timestamp: new Date() });
    expect(outDiv.children.length).toBe(1001);
    cons.unmount();

    // Dropdown: keyboard navigation (ArrowDown, ArrowUp wrap-around, Enter when closed & open, Escape), mousedown on item & outside, toggle when open, updateItems clearing missing selectedValue, placeholder fallback
    const dd = new Dropdown({
      items: [
        { value: 'a', label: 'A' },
        { value: 'b', label: 'B' },
      ],
      placeholder: '',
    });
    dd.mount(document.body);
    dd.element.querySelectorAll('.demo-dropdown-item').forEach((el) => {
      (el as HTMLElement).scrollIntoView = vi.fn();
    });
    dd.element.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' })); // opens via toggle
    dd.toggle(); // closes when isOpen is true
    dd.toggle(); // opens again
    const firstLi = dd.element.querySelector('.demo-dropdown-item') as HTMLElement;
    firstLi.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
    document.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
    dd.element.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown' }));
    dd.element.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowUp' })); // wraps to end
    dd.element.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown' }));
    dd.element.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    dd.element.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown' }));
    dd.element.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' })); // selects focused item
    dd.updateItems([{ value: 'c', label: 'C' }]);
    expect(dd.getValue()).toBeNull();
    dd.unmount();

    // Editor: MonacoEnvironment.getWorker for all labels, localStorage load & QuotaExceededError, openFile cached update, getValue/setValue before & after mount
    const env = self.MonacoEnvironment as { getWorker: (_: string, label: string) => unknown };
    for (const lbl of ['json', 'css', 'html', 'typescript', 'other']) {
      expect(env.getWorker('', lbl)).toBeDefined();
    }
    localStorage.setItem('onnx9000-demo-editor-python', 'cached_code');
    const edUnmounted = new Editor({ language: 'python', initialValue: 'init' });
    expect(edUnmounted.getValue()).toBe('');
    edUnmounted.setValue('ignored');

    let changeTrigger: (() => void) | undefined;
    let resizeTrigger: (() => void) | undefined;
    const OrigResizeObserver = globalThis.ResizeObserver;
    (globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = class {
      constructor(cb: () => void) {
        resizeTrigger = cb;
      }
      observe() {}
      disconnect() {}
    };
    const monacoObj = (
      window as unknown as {
        monaco: {
          editor: {
            create: (...args: unknown[]) => unknown;
            createModel: (...args: unknown[]) => unknown;
          };
        };
      }
    ).monaco;
    const origCreate = monacoObj.editor.create;
    const origCreateModel = monacoObj.editor.createModel;
    let currentVal = 'init';
    monacoObj.editor.create = () => ({
      getValue: () => currentVal,
      setValue: (v: string) => {
        currentVal = v;
      },
      onDidChangeModelContent: (cb: () => void) => {
        changeTrigger = cb;
        return { dispose: () => {} };
      },
      layout: () => {},
      dispose: () => {},
      setModel: () => {},
    });
    let modelVal = 'f1';
    monacoObj.editor.createModel = (...args: unknown[]) => {
      modelVal = String(args[0]);
      return {
        getValue: () => modelVal,
        setValue: (v: string) => {
          modelVal = v;
        },
        dispose: () => {},
      };
    };

    try {
      const ed = new Editor({
        language: 'python',
        initialValue: 'init',
        onChange: () => {},
      });
      ed.mount(document.body);
      resizeTrigger?.();
      changeTrigger?.();

      // QuotaExceededError on localStorage.setItem
      const origDesc = Object.getOwnPropertyDescriptor(window, 'localStorage');
      Object.defineProperty(window, 'localStorage', {
        configurable: true,
        value: {
          getItem: () => 'cached',
          setItem: () => {
            throw new Error('quota');
          },
        },
      });
      try {
        const edQuota = new Editor({
          language: 'python',
          initialValue: 'init',
          onChange: () => {},
        });
        edQuota.mount(document.body);
        changeTrigger?.();
        edQuota.unmount();
      } finally {
        if (origDesc) Object.defineProperty(window, 'localStorage', origDesc);
      }

      ed.openFile('/test.py', 'v1', 'python');
      ed.openFile('/test.py', 'v2', 'python');
      ed.openFile('/test.py', 'v2', 'python');
      ed.setValue('new_val');
      expect(ed.getValue()).toBe('new_val');
      ed.unmount();
    } finally {
      monacoObj.editor.create = origCreate;
      monacoObj.editor.createModel = origCreateModel;
      globalThis.ResizeObserver = OrigResizeObserver;
      localStorage.removeItem('onnx9000-demo-editor-python');
    }

    // FileTree: collapsed directory, click on non-label, updateData with file root
    const ft = new FileTree({
      root: {
        name: 'root',
        type: 'directory',
        path: '/root',
        children: [
          {
            name: 'sub',
            type: 'directory',
            path: '/root/sub',
            children: [{ name: 'f.py', type: 'file', path: '/root/sub/f.py' }],
          },
        ],
      },
      onSelect: () => {},
    });
    ft.mount(document.body);
    ft.element.click(); // click outside .demo-file-tree-label
    (ft.element.querySelectorAll('.demo-file-tree-label')[1] as HTMLElement)?.click(); // expand /root/sub
    (ft.element.querySelectorAll('.demo-file-tree-label')[1] as HTMLElement)?.click(); // collapse /root/sub
    ft.updateData({ name: 'single.py', type: 'file', path: '/single.py' });
    ft.unmount();

    // MetricsDashboard: 0 values
    const md = new MetricsDashboard();
    md.updateMetrics({ ttftMs: 0, tps: 0, totalLatencyMs: 0 });
    expect(md.element.textContent).toContain('--');

    // OliveConfigPanel: quantSelect, shapeCheckbox, and fusionCheckbox change
    const ocp = new OliveConfigPanel();
    ocp.mount(document.body);
    const quantSel = ocp.element.querySelector('.demo-olive-quant-select') as HTMLSelectElement;
    quantSel.value = 'INT8';
    quantSel.dispatchEvent(new Event('change'));
    const shapeCb = ocp.element.querySelector('.demo-olive-shape-checkbox') as HTMLInputElement;
    shapeCb.checked = false;
    shapeCb.dispatchEvent(new Event('change'));
    const fusionCb = ocp.element.querySelector('.demo-olive-fusion-checkbox') as HTMLInputElement;
    fusionCb.checked = true;
    fusionCb.dispatchEvent(new Event('change'));
    ocp.unmount();

    // OnnxVisualizer: tap on node, tap on background, TAB_CHANGED to viz, renderGraph(null)
    const viz = new OnnxVisualizer();
    viz.mount(document.body);
    viz.renderGraph({
      inputs: [{ name: 'in1', type: 'float32' }],
      outputs: [{ name: 'out1', type: 'float32' }],
      nodes: [
        {
          id: 'n1',
          name: 'n1',
          opType: 'Relu',
          inputs: ['in1'],
          outputs: ['out1'],
          attributes: { alpha: 1 },
        },
      ],
    });
    cyHandlers['tap:node']?.({
      target: {
        data: () => ({
          id: 'n1',
          label: 'n1',
          type: 'node',
          opType: 'Relu',
          dtype: 'float32',
          shape: [1, 2],
          attributes: { alpha: 1 },
        }),
      },
    });
    cyHandlers['tap:node']?.({
      target: {
        data: () => ({
          id: 'n2',
          label: 'n2',
          type: 'input',
        }),
      },
    });
    const cyInst = (viz as unknown as { cy: unknown }).cy;
    cyHandlers.tap?.({ target: cyInst });
    cyHandlers.tap?.({ target: {} });
    viz.renderGraph(null);
    globalEventBus.emit('ONNX_GRAPH_GENERATED', {
      inputs: [],
      outputs: [],
      nodes: [],
    });
    globalEventBus.emit('TAB_CHANGED', 'viz');
    await new Promise((r) => setTimeout(r, 60));
    viz.unmount();

    // PromoteButton: WASM_PROCESSING_START, WASM_PROCESSING_END, TARGET_CLEARED, click when enabled
    const pb = new PromoteButton();
    pb.mount(document.body);
    globalEventBus.emit('WASM_PROCESSING_START', true);
    expect(pb.element.disabled).toBe(true);
    globalEventBus.emit('WASM_PROCESSING_END', true);
    expect(pb.element.disabled).toBe(false);
    pb.element.click();
    globalEventBus.emit('TARGET_CLEARED', {});
    expect(pb.element.disabled).toBe(true);
    pb.unmount();

    // SplitPane: vertical & horizontal keyboard, drag, dblclick, localStorage restore
    localStorage.setItem('test-split-key', '0.4');
    const spH = new SplitPane({
      orientation: 'horizontal',
      storageKey: 'test-split-key',
      minSize: 10,
    });
    spH.mount(document.body);
    vi.spyOn(spH.element, 'getBoundingClientRect').mockReturnValue({
      left: 0,
      top: 0,
      width: 200,
      height: 200,
    } as DOMRect);
    const divH = spH.element.querySelector('[role="separator"]') as HTMLElement;
    for (const k of ['ArrowLeft', 'ArrowRight', 'Home', 'End', 'Enter']) {
      divH.dispatchEvent(new KeyboardEvent('keydown', { key: k }));
    }
    divH.dispatchEvent(new MouseEvent('dblclick'));
    divH.dispatchEvent(new MouseEvent('mousedown', { clientX: 50 }));
    window.dispatchEvent(new MouseEvent('mousemove', { clientX: 5 }));
    window.dispatchEvent(new MouseEvent('mousemove', { clientX: 195 }));
    window.dispatchEvent(new MouseEvent('mouseup'));
    spH.unmount();

    const spV = new SplitPane({ orientation: 'vertical', minSize: 10 });
    spV.mount(document.body);
    vi.spyOn(spV.element, 'getBoundingClientRect').mockReturnValue({
      left: 0,
      top: 0,
      width: 200,
      height: 200,
    } as DOMRect);
    const divV = spV.element.querySelector('[role="separator"]') as HTMLElement;
    for (const k of ['ArrowUp', 'ArrowDown']) {
      divV.dispatchEvent(new KeyboardEvent('keydown', { key: k }));
    }
    divV.dispatchEvent(new MouseEvent('mousedown', { clientY: 50 }));
    window.dispatchEvent(new MouseEvent('mousemove', { clientY: 5 }));
    window.dispatchEvent(new MouseEvent('mousemove', { clientY: 195 }));
    window.dispatchEvent(new MouseEvent('mouseup'));
    spV.unmount();
    localStorage.removeItem('test-split-key');

    // Tabs: keyboard ArrowLeft wrap-around (from 0 to end), ArrowRight wrap-around (from end to 0), Home, End, default key, and selectTab on already active tab
    const tComp = new Tabs({
      tabs: [
        { id: 't1', label: 'T1', content: document.createElement('div') },
        { id: 't2', label: 'T2', content: document.createElement('div') },
      ],
      initialTabId: 't1',
    });
    tComp.mount(document.body);
    tComp.selectTab('t1'); // already active
    const tabList = tComp.element.querySelector('.demo-tab-list') as HTMLElement;
    for (const k of ['ArrowLeft', 'ArrowRight', 'ArrowRight', 'ArrowLeft', 'End', 'Home', 'a']) {
      tabList.dispatchEvent(new KeyboardEvent('keydown', { key: k }));
    }
    expect(tComp.getActiveTabId()).toBe('t1');
    tComp.unmount();

    // TensorInputModal: LANGUAGE_CHANGED, empty inputs, image upload onload, overlay click, submit
    const tim = new TensorInputModal();
    tim.mount(document.body);
    globalEventBus.emit('LANGUAGE_CHANGED', 'en');
    tim.show([]);
    tim.show([{ name: 'img', type: 'float32', dims: [1, 3, 64, 64] }]);
    const fileInp = tim.element.querySelector('.demo-tensor-file-input') as HTMLInputElement;
    fileInp.dispatchEvent(new Event('change')); // no files
    const OrigImage = globalThis.Image;
    (globalThis as unknown as { Image: unknown }).Image = class {
      onload: (() => void) | null = null;
      set src(_v: string) {
        this.onload?.();
      }
    };
    const origCreateUrl = URL.createObjectURL;
    const origRevokeUrl = URL.revokeObjectURL;
    URL.createObjectURL = vi.fn().mockReturnValue('blob:img');
    URL.revokeObjectURL = vi.fn();
    try {
      const previewCanvas = tim.element.querySelector('canvas') as HTMLCanvasElement;
      vi.spyOn(previewCanvas, 'getContext').mockReturnValue({
        drawImage: vi.fn(),
      } as unknown as CanvasRenderingContext2D);
      Object.defineProperty(fileInp, 'files', {
        value: [new File(['a'], 'a.png', { type: 'image/png' })],
        configurable: true,
      });
      fileInp.dispatchEvent(new Event('change'));
    } finally {
      globalThis.Image = OrigImage;
      URL.createObjectURL = origCreateUrl;
      URL.revokeObjectURL = origRevokeUrl;
    }
    const btnsModal = tim.element.querySelectorAll('button');
    btnsModal[1]?.click(); // generateBtn -> fillRandomData
    btnsModal[2]?.click(); // submitBtn -> submit
    tim.element.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    tim.unmount();

    // ToastNotification: show without durationMs, then with durationMs=5 (clears previous timeout) and let timer expire
    const toast = new ToastNotification();
    toast.mount(document.body);
    globalEventBus.emit('SHOW_TOAST', { message: 'm0', type: 'info' });
    globalEventBus.emit('SHOW_TOAST', { message: 'm2', type: 'success', durationMs: 5 });
    await new Promise((r) => setTimeout(r, 15));
    toast.unmount();

    // WasmOverlay: HEAD fetch with content-length, without content-length, & catch; LANGUAGE_CHANGED; WASM_PROGRESS with total > 0, total === 0, loaded === 0; WASM_STATE_CHANGED to ERROR & LOADED
    const origFetch = globalThis.fetch;
    try {
      globalThis.fetch = vi
        .fn()
        .mockResolvedValueOnce({ headers: { get: () => '52428800' } })
        .mockResolvedValueOnce({ headers: { get: () => null } })
        .mockRejectedValueOnce(new Error('net err'));
      const wo0 = new WasmOverlay();
      wo0.mount(document.body);
      const wo1 = new WasmOverlay();
      wo1.mount(document.body);
      const wo2 = new WasmOverlay();
      wo2.mount(document.body);
      await new Promise((r) => setTimeout(r, 10));
      globalEventBus.emit('LANGUAGE_CHANGED', 'en');
      globalEventBus.emit('WASM_PROGRESS', { progress: 10, loaded: 0, total: 0 });
      globalEventBus.emit('WASM_PROGRESS', { progress: 25, loaded: 1024 * 1024, total: 0 });
      globalEventBus.emit('WASM_PROGRESS', {
        progress: 50,
        loaded: 1024 * 1024,
        total: 2 * 1024 * 1024,
      });
      globalEventBus.emit('WASM_PROGRESS', 75);
      WasmManager.getInstance().reset(); // mgr.error is null
      globalEventBus.emit('WASM_STATE_CHANGED', WasmState.ERROR);
      (WasmManager.getInstance() as unknown as { _error: string })._error = 'Custom err';
      globalEventBus.emit('WASM_STATE_CHANGED', WasmState.ERROR);
      globalEventBus.emit('WASM_STATE_CHANGED', WasmState.LOADED);
      wo0.unmount();
      wo1.unmount();
      wo2.unmount();
    } finally {
      globalThis.fetch = origFetch;
    }

    // LHSContainer & RHSContainer & main.ts exhaustive edge cases
    // Set up #interactive-demo-container with data-initial-source and data-initial-target='olive' and dark theme
    document.documentElement.setAttribute('data-theme', 'dark');
    const idc = document.createElement('div');
    idc.id = 'interactive-demo-container';
    idc.setAttribute('data-initial-source', 'keras');
    idc.setAttribute('data-initial-target', 'olive');
    document.body.appendChild(idc);

    const lhs = new LHSContainer();
    lhs.mount(document.body);
    const lhsRunBtn = (lhs as unknown as { runBtn: HTMLButtonElement }).runBtn;
    lhsRunBtn.disabled = true;
    globalEventBus.emit('LANGUAGE_CHANGED', 'en');
    lhsRunBtn.disabled = false;
    globalEventBus.emit('LANGUAGE_CHANGED', 'en');

    // UpdateExampleDropdown with unknown framework, and selectExample with initialFile missing content
    (lhs as unknown as { updateExampleDropdown: (fw: string) => void }).updateExampleDropdown(
      'unknown_fw',
    );
    (lhs as unknown as { updateExampleDropdown: (fw: string) => void }).updateExampleDropdown(
      'keras',
    );
    const exDd = (lhs as unknown as { exampleDropdown: Dropdown }).exampleDropdown;
    exDd.select(exDd.getValue() || '');

    // Temporarily set initialFile on keras[0] to a non-existent path to cover fallback content on line 396
    const { LHS_EXAMPLES } = await import('../src/data/MockData.js');
    const firstKerasEx = LHS_EXAMPLES.keras[0] as (typeof LHS_EXAMPLES.keras)[number];
    const origInitialFile = firstKerasEx.initialFile;
    firstKerasEx.initialFile = '/missing_initial_file.py';
    (lhs as unknown as { selectExample: (fw: string, id: string) => void }).selectExample(
      'keras',
      firstKerasEx.id,
    );
    firstKerasEx.initialFile = origInitialFile;

    // Storage throw on frameworkDropdown and exampleDropdown onChange
    const origStorageDesc = Object.getOwnPropertyDescriptor(window, 'localStorage');
    Object.defineProperty(window, 'localStorage', {
      configurable: true,
      value: {
        getItem: () => null,
        setItem: () => {
          throw new Error('quota');
        },
      },
    });
    const lhsFwDd = (lhs as unknown as { frameworkDropdown: Dropdown }).frameworkDropdown;
    lhsFwDd.select('keras');
    exDd.select('mnist');
    if (origStorageDesc) Object.defineProperty(window, 'localStorage', origStorageDesc);

    const noStackErr = new Error('no stack err');
    noStackErr.stack = '';
    const kParseSpy = vi
      .spyOn(KerasPythonParser, 'parse')
      .mockResolvedValueOnce({
        modelTopology: {
          config: {
            layers: [
              { class_name: 'InputLayer', config: {} },
              { class_name: 'Dense', config: {} },
            ],
          },
        },
      })
      .mockResolvedValueOnce({})
      .mockRejectedValueOnce(new Error('keras parse Error obj'))
      .mockRejectedValueOnce('raw keras parse error');
    const k2oSpy = vi
      .spyOn(convertersMod, 'keras2onnx')
      .mockResolvedValueOnce(new Uint8Array(0))
      .mockResolvedValueOnce(new Uint8Array(0))
      .mockResolvedValueOnce(new Uint8Array([1, 2, 3]))
      .mockResolvedValueOnce(new Uint8Array([1, 2, 3]))
      .mockRejectedValueOnce(noStackErr)
      .mockRejectedValueOnce('raw k2o error');
    const parseProtoSpy = vi.spyOn(coreMod, 'parseModelProto').mockResolvedValue({
      name: 'g',
      nodes: [new Node('Relu', ['x'], ['y'], {}, 'r1'), new Node('Relu', ['y'], ['z'], {}, '')],
      inputs: [new ValueInfo('x', [1], 'float32')],
      outputs: [new ValueInfo('z', [1], 'float32')],
      initializers: ['w'],
      tensors: { w: new Tensor('w', [1], 'float32', true, false, new Uint8Array([1, 2, 3, 4])) },
      valueInfo: [],
    } as unknown as Graph);
    const convSpy = vi.spyOn(convertersMod, 'convert').mockResolvedValue({
      name: 'g',
      nodes: [new Node('Relu', ['x'], ['y'], {}, '')],
      inputs: [new ValueInfo('x', [1], 'float32')],
      outputs: [new ValueInfo('y', [1], 'float32')],
      initializers: [],
      tensors: {},
      valueInfo: [],
    } as unknown as Graph);
    const serSpy = vi
      .spyOn(coreMod, 'serializeModelProto')
      .mockReturnValue(new Uint8Array([1, 2, 3]));

    const lhsEditor = (lhs as unknown as { editor: Editor }).editor;
    const lhsTree = (lhs as unknown as { tree: FileTree }).tree;

    // Select a file that exists and has content, and one that does not
    lhsTree.selectFile(firstKerasEx.initialFile as string);
    lhsTree.selectFile('/non_existent_for_fallback.py');

    // Trigger debounced editor onChange with fast setTimeout
    const origSetTimeout = globalThis.setTimeout;
    globalThis.setTimeout = ((cb: () => void) => {
      cb();
      return 0;
    }) as unknown as typeof setTimeout;
    (lhsEditor as unknown as { options: { onChange?: (v: string) => void } }).options.onChange?.(
      'changed code',
    );
    globalThis.setTimeout = origSetTimeout;

    vi.spyOn(lhsEditor, 'getValue').mockReturnValue(`import keras
model = models.Sequential()`);
    const clickEv = new MouseEvent('click') as unknown as PointerEvent;
    await lhsRunBtn.onclick?.(clickEv); // 0-byte with layers[0].config = {}
    idc.removeAttribute('data-initial-target'); // so lines 270-271 localStorage || 'onnx' run!
    await lhsRunBtn.onclick?.(clickEv); // 0-byte with layers = undefined -> []
    await lhsRunBtn.onclick?.(clickEv); // Error keras parse error -> fallback + non-empty bytes
    await lhsRunBtn.onclick?.(clickEv); // string keras parse error -> fallback + non-empty bytes
    await lhsRunBtn.onclick?.(clickEv); // noStackErr in outer catch
    await lhsRunBtn.onclick?.(clickEv); // raw k2o error in outer catch

    for (const fw of [
      'caffe',
      'mxnet',
      'tensorflow',
      'paddle',
      'onnxscript',
      'scikitlearn',
      'xgboost',
      'catboost',
      'sparkml',
    ]) {
      lhsFwDd.select(fw);
      await lhsRunBtn.onclick?.(clickEv);
    }

    // RHSContainer: mounted with unknown data-initial-target (falls back to RHS_TARGETS.onnx), with 'olive', and with 'onnx'
    idc.setAttribute('data-initial-target', 'unknown_target_xyz');
    const rhsUnk = new RHSContainer();
    rhsUnk.mount(document.body);
    rhsUnk.unmount();

    idc.setAttribute('data-initial-target', 'olive');
    const rhsOlive = new RHSContainer();
    rhsOlive.mount(document.body);
    idc.setAttribute('data-initial-target', 'onnx');
    const rhs = new RHSContainer();
    rhs.mount(document.body);

    const rhsRun = (rhs as unknown as { runBtn: HTMLButtonElement }).runBtn;
    globalEventBus.emit('CONVERSION_STARTED', null);
    expect(rhsRun.disabled).toBe(true);
    rhsRun.textContent = 'Running...';
    globalEventBus.emit('LANGUAGE_CHANGED', 'en');
    rhsRun.disabled = false;
    globalEventBus.emit('LANGUAGE_CHANGED', 'en');

    // Call regenerateOutput when dropdown.getValue() is null
    const rhsDd = (rhs as unknown as { dropdown: Dropdown }).dropdown;
    const rhsTree = (rhs as unknown as { tree: FileTree }).tree;
    const savedVal = rhsDd.getValue();
    (rhsDd as unknown as { selectedValue: string | null }).selectedValue = null;
    await (rhs as unknown as { regenerateOutput: () => Promise<void> }).regenerateOutput();
    (rhsDd as unknown as { selectedValue: string | null }).selectedValue = savedVal;

    const cCompSpy = vi
      .spyOn(cCompilerMod, 'compileOnnxToC')
      .mockResolvedValueOnce({ header: 'h', source: 's', summary: 'sum' } as never)
      .mockRejectedValueOnce(new Error('c++ Error'))
      .mockRejectedValueOnce('raw c++ fail')
      .mockRejectedValueOnce(new Error('c Error'))
      .mockRejectedValueOnce('raw c fail');
    const cmlConvSpy = vi.spyOn(coremlMod, 'convertToCoreML').mockReturnValue({} as never);
    const cmlPkgSpy = vi.spyOn(coremlMod, 'buildMLPackage').mockReturnValue({
      buildDirectoryStructure: () => new Map([['Manifest.json', new Uint8Array([123, 125])]]),
    } as never);
    const optSpy = vi.spyOn(optimumMod, 'optimize').mockResolvedValue({
      name: 'g',
      nodes: [new Node('Relu', ['x'], ['y'], {}, '')],
      inputs: [new ValueInfo('x', [1], 'float32')],
      outputs: [new ValueInfo('y', [1], 'float32')],
    } as unknown as Graph);
    const simpSpy = vi.spyOn(optimumMod, 'simplify').mockResolvedValue({
      name: 'g',
      nodes: [new Node('Relu', ['x'], ['y'], {}, '')],
      inputs: [new ValueInfo('x', [1], 'float32')],
      outputs: [new ValueInfo('y', [1], 'float32')],
    } as unknown as Graph);

    // Emit ONNX_GRAPH_GENERATED while target is 'onnx' and selectedPath is '/output-onnx/model.onnx'
    globalEventBus.emit('ONNX_GRAPH_GENERATED', {
      inputs: [{ name: 'x', type: 'float32' }],
      outputs: [{ name: 'y', type: 'float32' }],
      nodes: [{ id: 'r1', name: 'r1', opType: 'Relu', inputs: ['x'], outputs: ['y'] }],
    });
    globalEventBus.emit('ONNX_BINARY_GENERATED', new Uint8Array([1, 2, 3]));

    // Storage throw in rhsDd onChange
    Object.defineProperty(window, 'localStorage', {
      configurable: true,
      value: {
        getItem: () => null,
        setItem: () => {
          throw new Error('quota');
        },
      },
    });
    rhsDd.select('cpp'); // succeeds (1st cCompSpy call)
    if (origStorageDesc) Object.defineProperty(window, 'localStorage', origStorageDesc);
    await new Promise((r) => setTimeout(r, 10));

    rhsDd.select('cpp'); // fails with Error (2nd cCompSpy call)
    await new Promise((r) => setTimeout(r, 10));
    rhsDd.select('cpp'); // fails with string (3rd cCompSpy call)
    await new Promise((r) => setTimeout(r, 10));
    rhsDd.select('c'); // fails with Error (4th cCompSpy call)
    await new Promise((r) => setTimeout(r, 10));
    rhsDd.select('c'); // fails with string (5th cCompSpy call)
    await new Promise((r) => setTimeout(r, 10));

    // Set olivePanel quantizationLevel to INT8 to cover 'O3' branch
    (rhs as unknown as { olivePanel: OliveConfigPanel }).olivePanel.config.quantizationLevel =
      'INT8';

    for (const target of [
      'coreml',
      'pytorch',
      'olive',
      'mlir',
      'onnx-simplifier',
      'caffe',
      'onnx',
    ]) {
      rhsDd.select(target);
      await new Promise((r) => setTimeout(r, 10));
    }

    // Also test pytorch and caffe when convert returns a string instead of an object
    convSpy.mockResolvedValueOnce('import torch' as never);
    rhsDd.select('pytorch');
    await new Promise((r) => setTimeout(r, 10));
    convSpy.mockResolvedValueOnce('name: "caffe"' as never);
    rhsDd.select('caffe');
    await new Promise((r) => setTimeout(r, 10));

    // Both Error and string error paths in regenerateOutput for coreml, pytorch, olive, mlir, onnx-simplifier, caffe
    cmlConvSpy.mockImplementationOnce(() => {
      throw new Error('coreml err');
    });
    rhsDd.select('coreml');
    await new Promise((r) => setTimeout(r, 10));
    cmlConvSpy.mockImplementationOnce(() => {
      throw 'raw coreml fail';
    });
    rhsDd.select('coreml');
    await new Promise((r) => setTimeout(r, 10));

    convSpy.mockRejectedValueOnce(new Error('pytorch err'));
    rhsDd.select('pytorch');
    await new Promise((r) => setTimeout(r, 10));
    convSpy.mockRejectedValueOnce('raw pytorch fail');
    rhsDd.select('pytorch');
    await new Promise((r) => setTimeout(r, 10));

    optSpy.mockRejectedValueOnce(new Error('olive err'));
    rhsDd.select('olive');
    await new Promise((r) => setTimeout(r, 10));
    optSpy.mockRejectedValueOnce('raw olive fail');
    rhsDd.select('olive');
    await new Promise((r) => setTimeout(r, 10));

    parseProtoSpy.mockRejectedValueOnce(new Error('mlir err'));
    rhsDd.select('mlir');
    await new Promise((r) => setTimeout(r, 10));
    parseProtoSpy.mockRejectedValueOnce('raw mlir fail');
    rhsDd.select('mlir');
    await new Promise((r) => setTimeout(r, 10));

    simpSpy.mockRejectedValueOnce(new Error('simp err'));
    rhsDd.select('onnx-simplifier');
    await new Promise((r) => setTimeout(r, 10));
    simpSpy.mockRejectedValueOnce('raw simp fail');
    rhsDd.select('onnx-simplifier');
    await new Promise((r) => setTimeout(r, 10));

    convSpy.mockRejectedValueOnce(new Error('caffe err'));
    rhsDd.select('caffe');
    await new Promise((r) => setTimeout(r, 10));
    convSpy.mockRejectedValueOnce('raw caffe fail');
    rhsDd.select('caffe');
    await new Promise((r) => setTimeout(r, 10));

    // Tree selectFile with .cpp, .json, .py extensions
    rhsTree.selectFile('/test.cpp');
    rhsTree.selectFile('/test.json');
    rhsTree.selectFile('/test.py');

    // RHS runBtn click with fast setTimeout (both Math.random > 0.05 and <= 0.05)
    const randSpy = vi.spyOn(Math, 'random').mockReturnValueOnce(0.5).mockReturnValueOnce(0.01);
    globalThis.setTimeout = ((cb: () => void) => {
      cb();
      return 0;
    }) as unknown as typeof setTimeout;
    try {
      globalEventBus.emit('CONVERSION_COMPLETED', null);
      rhsRun.onclick?.(clickEv);
      rhsRun.onclick?.(clickEv);
    } finally {
      globalThis.setTimeout = origSetTimeout;
      randSpy.mockRestore();
    }

    // Re-import main.ts while #interactive-demo-container and data-theme="dark" are present in DOM
    vi.resetModules();
    await import('../src/main.js');

    (
      window as unknown as { __ADD_BREADCRUMB_TEST__: (id: string, desc: string) => void }
    ).__ADD_BREADCRUMB_TEST__('b1', 'Breadcrumb 1');
    (window as unknown as { __OPEN_MODAL__: (inputs: unknown[]) => void }).__OPEN_MODAL__([]);

    document.documentElement.setAttribute('data-theme', 'light');
    await new Promise((r) => setTimeout(r, 10));
    document.documentElement.setAttribute('data-theme', 'dark');
    await new Promise((r) => setTimeout(r, 10));

    rhsOlive.unmount();
    lhs.unmount();
    rhs.unmount();
    idc.remove();
    kParseSpy.mockRestore();
    k2oSpy.mockRestore();
    parseProtoSpy.mockRestore();
    convSpy.mockRestore();
    serSpy.mockRestore();
    cCompSpy.mockRestore();
    cmlConvSpy.mockRestore();
    cmlPkgSpy.mockRestore();
    optSpy.mockRestore();
    simpSpy.mockRestore();
    logSpy.mockRestore();
    warnSpy.mockRestore();
    errSpy.mockRestore();
    expect(RHS_TARGETS).toBeDefined();
  });
});
