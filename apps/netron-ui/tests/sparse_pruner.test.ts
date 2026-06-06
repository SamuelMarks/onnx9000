import { describe, it, expect, vi, beforeEach } from 'vitest';
import { SparsePrunerUI } from '../src/sparse_pruner.ts';

vi.mock('@onnx9000/core', () => {
  return {
    Graph: class {
      tensors: Record<string, any> = {};
      constructor(name: string) {}
    },
    unpackData: vi.fn().mockReturnValue([0.1, -0.2, 0.0, 0.5]),
  };
});

vi.mock('@onnx9000/modifier', () => {
  return {
    applyRecipe: vi.fn(),
  };
});

describe('SparsePrunerUI', () => {
  beforeEach(() => {
    document.body.innerHTML = '<div id="log"></div><input type="range" id="sparsity-slider" value="50" /><div id="sparsity-value"></div><button id="run-btn"></button><div id="drop-zone"></div><div id="param-count"></div><div id="current-sparsity"></div><div id="est-speedup"></div><div id="progress" style="display:none;"></div><div id="progress-fill"></div><div id="progress-text"></div><button id="download-btn" disabled></button>';
    vi.restoreAllMocks();
  });

  it('should initialize correctly', () => {
    const ui = new SparsePrunerUI();
    expect(ui).toBeDefined();
  });

  it('should handle drag and drop models', async () => {
    const ui = new SparsePrunerUI();
    const dropZone = document.getElementById('drop-zone')!;
    const mockFile = new File(['mock content'], 'model.onnx', { type: 'application/octet-stream' });
    const dropEvent = new Event('drop') as any;
    dropEvent.dataTransfer = { files: [mockFile] };
    dropZone.dispatchEvent(dropEvent);
    
    // Wait for async loadModel
    await new Promise(r => setTimeout(r, 0));
    
    // Fallback to textContent if innerText is undefined in JSDOM
    const el = document.getElementById('param-count')!;
    expect(el.innerText || el.textContent).toBe('1.2M');
  });

  it('should handle loadModel', async () => {
    const ui = new SparsePrunerUI();
    await ui.loadModel(new Uint8Array([1, 2, 3]));
    const el = document.getElementById('param-count')!;
    expect(el.innerText || el.textContent).toBe('1.2M');
  });

  it('should handle runPruning', async () => {
    const ui = new SparsePrunerUI();
    await ui.loadModel(new Uint8Array([1, 2, 3]));
    const runPromise = ui.runPruning();
    await runPromise;
    expect((document.getElementById('download-btn') as HTMLButtonElement).disabled).toBe(false);
  });
});
