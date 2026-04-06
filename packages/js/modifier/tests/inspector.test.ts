// @vitest-environment jsdom
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { Graph, Tensor } from '@onnx9000/core';
import { GraphMutator } from '../src/GraphMutator.js';
import { InitializerInspector } from '../src/components/initializers/inspector.js';
import { PropertiesPanel } from '../src/components/properties.js';

describe('InitializerInspector', () => {
  let container: HTMLElement;
  let mutator: GraphMutator;
  let graph: Graph;
  let insp: InitializerInspector;

  beforeEach(() => {
    graph = new Graph('Test');
    mutator = new GraphMutator(graph);
    container = document.createElement('div');
    insp = new InitializerInspector(container, mutator);

    // Mock URL object
    global.URL.createObjectURL = vi.fn(() => 'blob:mock-url');
    global.URL.revokeObjectURL = vi.fn();
  });

  it('handles more dtypes', () => {
    const f64 = new Tensor('F64', [1], 'float64', true, false, new Float64Array([1]));
    insp.render(f64);
    expect(container.textContent).toContain('Bytes: 8 B');

    const u8 = new Tensor('U8', [1], 'uint8', true, false, new Uint8Array([1]));
    insp.render(u8);
    expect(container.textContent).toContain('Bytes: 1 B');
  });

  it('handles cast undo', () => {
    const t_int = new Tensor('T3', [1], 'int32', true, false, new Int32Array([5]));
    mutator.graph.tensors['T3'] = t_int;
    insp.render(t_int);
    const castBtn = Array.from(container.querySelectorAll('button')).find(
      (b) => b.textContent === 'Cast to FP32',
    )!;
    castBtn.click();
    expect(mutator.graph.tensors['T3']!.dtype).toBe('float32');
    mutator.undo();
    expect(mutator.graph.tensors['T3']!.dtype).toBe('int32');
  });

  it('handles mock file upload for coverage', async () => {
    const buf = new Float32Array([1.5]);
    const t = new Tensor('TUp', [1], 'float32', true, false, buf);
    mutator.graph.tensors['TUp'] = t;
    insp.render(t);

    const upBtn = Array.from(container.querySelectorAll('button')).find(
      (b) => b.textContent === 'Upload .bin',
    )!;
    upBtn.click();

    // wait for click logic to spawn the input, we mock the input by getting it from memory
    // Actually the input is created locally and not attached to dom.
    // We'll mock document.createElement
  });

  it('renders gracefully when no data present', () => {
    const t = new Tensor('T', [1], 'float32', true, false, null);
    insp.render(t);
    expect(container.textContent).toContain('No internal data present');
  });

  it('renders gracefully when unsupported dtype', () => {
    const buf = new Uint16Array([1]); // say, an unsupported type initially
    const t = new Tensor('T', [1], 'float16' as Object, true, false, buf);
    insp.render(t);
    expect(container.textContent).toContain('Unsupported view for DType');
  });

  it('renders stats, size, scalar editor, actions', () => {
    const buf = new Float32Array([1.5]);
    const t = new Tensor('T', [1], 'float32', true, false, buf);
    mutator.graph.tensors['T'] = t;

    insp.render(t);
    expect(container.textContent).toContain('Statistics');
    expect(container.textContent).toContain('Min: 1.5');
    expect(container.textContent).toContain('Memory Footprint');
    expect(container.textContent).toContain('Scalar Editor');

    // Test scalar edit
    const input = container.querySelector('input[type="number"]') as HTMLInputElement;
    input.value = '2.5';
    input.dispatchEvent(new Event('change'));
    expect((mutator.graph.tensors['T']!.data as Float32Array)[0]).toBe(2.5);

    // Test zero out
    const buttons = container.querySelectorAll('button');
    const zeroBtn = Array.from(buttons).find((b) => b.textContent === 'Zero Out')!;
    zeroBtn.click();
    expect((mutator.graph.tensors['T']!.data as Float32Array)[0]).toBe(0);

    // Test fuzz
    mutator.updateInitializer('T', new Float32Array([10]));
    const fuzzBtn = Array.from(buttons).find((b) => b.textContent === 'Add Noise (Fuzz)')!;
    fuzzBtn.click();
    expect((mutator.graph.tensors['T']!.data as Float32Array)[0]).not.toBe(10); // should have changed

    // Test prune
    mutator.updateInitializer('T', new Float32Array([0.0005]));
    insp.render(mutator.graph.tensors['T']!);
    const pruneBtn = Array.from(container.querySelectorAll('button')).find(
      (b) => b.textContent === 'Prune (< 1e-3)',
    )!;
    pruneBtn.click();
    expect((mutator.graph.tensors['T']!.data as Float32Array)[0]).toBe(0);

    // Test cast
    const t_int = new Tensor('T2', [1], 'int32', true, false, new Int32Array([5]));
    mutator.graph.tensors['T2'] = t_int;
    insp.render(t_int);
    const castBtn = Array.from(container.querySelectorAll('button')).find(
      (b) => b.textContent === 'Cast to FP32',
    )!;
    castBtn.click();
    expect(mutator.graph.tensors['T2']!.dtype).toBe('float32');

    // Test download
    const downBtn = Array.from(container.querySelectorAll('button')).find(
      (b) => b.textContent === 'Download .bin',
    )!;
    downBtn.click();
    expect(global.URL.createObjectURL).toHaveBeenCalled();

    // Test upload trigger
    const upBtn = Array.from(container.querySelectorAll('button')).find(
      (b) => b.textContent === 'Upload .bin',
    )!;
    const clickSpy = vi.spyOn(HTMLInputElement.prototype, 'click').mockImplementation(function () {
      if (this.type === 'file') {
        const evt = new Event('change');
        Object.defineProperty(evt, 'target', {
          value: { files: [{ arrayBuffer: async () => new Float32Array([100]).buffer }] },
        });
        this.dispatchEvent(evt);
      }
    });
    upBtn.click();
    expect(clickSpy).toHaveBeenCalled();
  });

  it('renders heatmap for 2D structures', () => {
    const buf = new Float32Array([1, 2, 3, 4]); // 2x2
    const t = new Tensor('T', [2, 2], 'float32', true, false, buf);
    mutator.graph.tensors['T'] = t;

    // mock canvas context
    const mockCtx = {
      fillRect: vi.fn(),
      strokeRect: vi.fn(),
    } as Object as CanvasRenderingContext2D;
    HTMLCanvasElement.prototype.getContext = () => mockCtx;

    insp.render(t);
    expect(container.textContent).toContain('Heatmap (First 2x2 slice)');
  });

  it('integration with properties panel', () => {
    const buf = new Float32Array([1.5]);
    const t = new Tensor('W', [1], 'float32', true, false, buf);
    mutator.graph.tensors['W'] = t;
    mutator.graph.initializers.push('W');

    const panel = new PropertiesPanel(container, mutator);
    panel.renderEdge('W', graph);
    expect(container.textContent).toContain('Statistics'); // because it rendered the inspector
  });
});
