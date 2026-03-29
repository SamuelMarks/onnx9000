/* eslint-disable */
// @ts-nocheck
import { describe, it, expect, vi } from 'vitest';
import { RHSContainer } from '../../src/components/RHSContainer';
import { RHS_TARGETS } from '../../src/data/MockData';
import { globalEventBus } from '../../src/core/EventBus';
import { compileOnnxToC } from '@onnx9000/c-compiler';

// Mock components so we can test the container logic
vi.mock('../../src/components/Dropdown', () => ({
  Dropdown: class {
    options: object;
    _value: object;
    constructor(options: object) {
      this.options = options;
      this._value = options.initialValue || null;
    }
    mount() {}
    triggerChange(val: string) {
      this._value = val;
      if (this.options.onChange) this.options.onChange(val);
    }
    getValue() {
      return this._value;
    }
  }
}));

vi.mock('../../src/components/FileTree', () => ({
  FileTree: class {
    options: object;
    constructor(options: object) {
      this.options = options;
    }
    mount() {}
    updateData = vi.fn();
    findNode = vi.fn().mockReturnValue(null);
    getSelectedPath = vi.fn().mockReturnValue('');
    selectFile = vi.fn();
    triggerSelect(path: string) {
      if (this.options.onSelect) this.options.onSelect(path);
    }
  }
}));

vi.mock('../../src/components/Editor', () => ({
  Editor: class {
    mount() {}
    openFile = vi.fn();
  }
}));

vi.mock('../../src/components/SplitPane', () => ({
  SplitPane: class {
    mount() {}
    getPanes() {
      return { pane1: document.createElement('div'), pane2: document.createElement('div') };
    }
  }
}));

vi.mock('@onnx9000/c-compiler', () => ({
  compileOnnxToC: vi.fn().mockResolvedValue({
    header: '// mocked header',
    source: '// mocked source'
  })
}));

vi.mock('@onnx9000/converters', () => ({
  convert: vi.fn().mockResolvedValue('def forward(): pass')
}));

vi.mock('@onnx9000/optimum', () => ({
  optimize: vi.fn().mockResolvedValue({
    nodes: [],
    inputs: [],
    outputs: [],
    initializers: [],
    tensors: {},
    valueInfo: []
  }),
  simplify: vi.fn().mockResolvedValue({
    nodes: [],
    inputs: [],
    outputs: [],
    initializers: [],
    tensors: {},
    valueInfo: []
  })
}));

vi.mock('@onnx9000/core', () => ({
  BufferReader: class {},
  parseModelProto: vi.fn().mockResolvedValue({
    nodes: [],
    inputs: [],
    outputs: [],
    initializers: [],
    tensors: {},
    valueInfo: []
  })
}));

vi.mock('@onnx9000/iree-compiler/dist/passes/lower_onnx_to_mhlo.js', () => ({
  lowerONNXToMHLO: vi.fn().mockReturnValue({})
}));

vi.mock('@onnx9000/iree-compiler/dist/passes/interop.js', () => ({
  MLIRInterop: class {
    emitMLIR = vi.fn().mockReturnValue('module {}');
  }
}));

vi.mock('@onnx9000/coreml', () => ({
  convertToCoreML: vi.fn().mockReturnValue({}),
  buildMLPackage: vi.fn().mockReturnValue({
    buildDirectoryStructure: vi
      .fn()
      .mockReturnValue(new Map([['Manifest.json', new Uint8Array([123, 125])]]))
  })
}));

describe('RHSContainer', () => {
  it('should render and mount components', () => {
    const container = new RHSContainer();
    const el = (container as object).element as HTMLElement;
    expect(el.className).toBe('demo-pane-rhs');
    container.unmount();
  });

  it('should update tree data when dropdown changes', () => {
    const container = new RHSContainer();

    // Simulate dropdown change
    const dropdown = (container as object).dropdown;
    const tree = (container as object).tree;

    dropdown.triggerChange('mlir');
    expect(tree.updateData).toHaveBeenCalledWith(RHS_TARGETS['mlir']);
    container.unmount();
  });

  it('should open file in editor when tree item selected', () => {
    const container = new RHSContainer();

    const tree = (container as object).tree;
    const editor = (container as object).editor;

    tree.triggerSelect('/output-mlir/graph.mlir');
    expect(editor.openFile).toHaveBeenCalledWith(
      '/output-mlir/graph.mlir',
      '// Binary representation of /output-mlir/graph.mlir',
      'plaintext'
    );
    container.unmount();
  });

  it('should compile C++ code when ONNX_BINARY_GENERATED is fired and target is cpp', async () => {
    const container = new RHSContainer();
    container.mount(document.body);
    const dropdown = (container as object).dropdown;
    const editor = (container as object).editor;
    const tree = (container as object).tree;

    dropdown.triggerChange('cpp');
    tree.findNode.mockReturnValue({ content: '' });
    tree.getSelectedPath.mockReturnValue('/output-cpp/model.cpp');

    globalEventBus.emit('ONNX_BINARY_GENERATED', new Uint8Array([1, 2, 3]));
    await new Promise((r) => setTimeout(r, 10)); // wait for promise resolution
    expect(compileOnnxToC).toHaveBeenCalledWith(new Uint8Array([1, 2, 3]), {
      prefix: 'model_',
      emitCpp: true
    });
    expect((container as object).tree.selectFile).toHaveBeenCalledWith('/output-cpp/model.cpp');
    container.unmount();
  });

  it('should compile CoreML when ONNX_BINARY_GENERATED is fired and target is coreml', async () => {
    const container = new RHSContainer();
    container.mount(document.body);
    const dropdown = (container as object).dropdown;
    const editor = (container as object).editor;
    const tree = (container as object).tree;

    dropdown.triggerChange('coreml');
    tree.findNode.mockReturnValue({ content: '' });
    tree.getSelectedPath.mockReturnValue('/model.mlpackage/Manifest.json');

    globalEventBus.emit('ONNX_BINARY_GENERATED', new Uint8Array([1, 2, 3]));
    await new Promise((r) => setTimeout(r, 10)); // wait for promise resolution
    expect((container as object).tree.selectFile).toHaveBeenCalledWith(
      '/model.mlpackage/Manifest.json'
    );
    container.unmount();
  });

  it('should compile PyTorch when ONNX_BINARY_GENERATED is fired and target is pytorch', async () => {
    const container = new RHSContainer();
    container.mount(document.body);
    const dropdown = (container as object).dropdown;
    const editor = (container as object).editor;
    const tree = (container as object).tree;

    dropdown.triggerChange('pytorch');
    tree.findNode.mockReturnValue({ content: '' });
    tree.getSelectedPath.mockReturnValue('/output-pytorch/module.py');

    globalEventBus.emit('ONNX_BINARY_GENERATED', new Uint8Array([1, 2, 3]));
    await new Promise((r) => setTimeout(r, 10)); // wait for promise resolution
    expect((container as object).tree.selectFile).toHaveBeenCalledWith('/output-pytorch/module.py');
    container.unmount();
  });

  it('should optimize Olive when ONNX_BINARY_GENERATED is fired and target is olive', async () => {
    const container = new RHSContainer();
    container.mount(document.body);
    const dropdown = (container as object).dropdown;
    const editor = (container as object).editor;
    const tree = (container as object).tree;

    dropdown.triggerChange('olive');
    tree.findNode.mockReturnValue({ content: '' });
    tree.getSelectedPath.mockReturnValue('/olive-optimized/optimized_model.onnx');

    // It parses the graph, OnnxAstFormatter format returns '' by default when mocking missing methods, but we just verify it opens file
    globalEventBus.emit('ONNX_BINARY_GENERATED', new Uint8Array([1, 2, 3]));
    await new Promise((r) => setTimeout(r, 10)); // wait for promise resolution
    expect((container as object).tree.selectFile).toHaveBeenCalledWith(
      '/olive-optimized/optimized_model.onnx'
    );
    container.unmount();
  });

  it('should generate MLIR when ONNX_BINARY_GENERATED is fired and target is mlir', async () => {
    const container = new RHSContainer();
    container.mount(document.body);
    const dropdown = (container as object).dropdown;
    const editor = (container as object).editor;
    const tree = (container as object).tree;

    dropdown.triggerChange('mlir');
    tree.findNode.mockReturnValue({ content: '' });
    tree.getSelectedPath.mockReturnValue('/output-mlir/graph.mlir');

    globalEventBus.emit('ONNX_BINARY_GENERATED', new Uint8Array([1, 2, 3]));
    await new Promise((r) => setTimeout(r, 10)); // wait for promise resolution
    expect((container as object).tree.selectFile).toHaveBeenCalledWith('/output-mlir/graph.mlir');
    container.unmount();
  });

  it('should simplify when ONNX_BINARY_GENERATED is fired and target is onnx-simplifier', async () => {
    const container = new RHSContainer();
    container.mount(document.body);
    const dropdown = (container as object).dropdown;
    const editor = (container as object).editor;
    const tree = (container as object).tree;

    dropdown.triggerChange('onnx-simplifier');
    tree.findNode.mockReturnValue({ content: '' });
    tree.getSelectedPath.mockReturnValue('/simplified-model/simplified.onnx');

    globalEventBus.emit('ONNX_BINARY_GENERATED', new Uint8Array([1, 2, 3]));
    await new Promise((r) => setTimeout(r, 10));
    expect((container as object).tree.selectFile).toHaveBeenCalledWith(
      '/simplified-model/simplified.onnx'
    );
    container.unmount();
  });

  it('should convert to arbitrary frameworks when ONNX_BINARY_GENERATED is fired and target is caffe', async () => {
    const container = new RHSContainer();
    container.mount(document.body);
    const dropdown = (container as object).dropdown;
    const editor = (container as object).editor;
    const tree = (container as object).tree;

    dropdown.triggerChange('caffe');
    tree.findNode.mockReturnValue({ content: '' });
    tree.getSelectedPath.mockReturnValue('/output-caffe/model.prototxt');

    globalEventBus.emit('ONNX_BINARY_GENERATED', new Uint8Array([1, 2, 3]));
    await new Promise((r) => setTimeout(r, 10));
    expect((container as object).tree.selectFile).toHaveBeenCalledWith(
      '/output-caffe/model.prototxt'
    );
    container.unmount();
  });

  it('should handle C++ compilation failure gracefully', async () => {
    const container = new RHSContainer();
    container.mount(document.body);
    const dropdown = (container as object).dropdown;
    dropdown.triggerChange('cpp');

    const { compileOnnxToC } = await import('@onnx9000/c-compiler');
    (compileOnnxToC as object).mockRejectedValueOnce(new Error('C++ error'));

    const consoleSpy = vi.spyOn(console, 'error');
    globalEventBus.emit('ONNX_BINARY_GENERATED', new Uint8Array([1, 2, 3]));
    await new Promise((r) => setTimeout(r, 10));
    expect(consoleSpy).toHaveBeenCalledWith(
      expect.stringContaining('C++ compilation failed: C++ error')
    );
    consoleSpy.mockRestore();
    container.unmount();
  });

  it('should handle CoreML compilation failure gracefully', async () => {
    const container = new RHSContainer();
    container.mount(document.body);
    const dropdown = (container as object).dropdown;
    dropdown.triggerChange('coreml');

    const { convertToCoreML } = await import('@onnx9000/coreml');
    (convertToCoreML as object).mockImplementationOnce(() => {
      throw new Error('CoreML error');
    });

    const consoleSpy = vi.spyOn(console, 'error');
    globalEventBus.emit('ONNX_BINARY_GENERATED', new Uint8Array([1, 2, 3]));
    await new Promise((r) => setTimeout(r, 10));
    expect(consoleSpy).toHaveBeenCalledWith(
      expect.stringContaining('CoreML compilation failed: CoreML error')
    );
    consoleSpy.mockRestore();
    container.unmount();
  });

  it('should handle PyTorch conversion failure gracefully', async () => {
    const container = new RHSContainer();
    container.mount(document.body);
    const dropdown = (container as object).dropdown;
    dropdown.triggerChange('pytorch');

    const { convert } = await import('@onnx9000/converters');
    (convert as object).mockRejectedValueOnce(new Error('PyTorch error'));

    const consoleSpy = vi.spyOn(console, 'error');
    globalEventBus.emit('ONNX_BINARY_GENERATED', new Uint8Array([1, 2, 3]));
    await new Promise((r) => setTimeout(r, 10));
    expect(consoleSpy).toHaveBeenCalledWith(
      expect.stringContaining('PyTorch conversion failed: PyTorch error')
    );
    consoleSpy.mockRestore();
    container.unmount();
  });

  it('should handle Olive optimization failure gracefully', async () => {
    const container = new RHSContainer();
    container.mount(document.body);
    const dropdown = (container as object).dropdown;
    dropdown.triggerChange('olive');

    const { optimize } = await import('@onnx9000/optimum');
    (optimize as object).mockRejectedValueOnce(new Error('Olive error'));

    const consoleSpy = vi.spyOn(console, 'error');
    globalEventBus.emit('ONNX_BINARY_GENERATED', new Uint8Array([1, 2, 3]));
    await new Promise((r) => setTimeout(r, 10));
    expect(consoleSpy).toHaveBeenCalledWith(
      expect.stringContaining('Olive optimization failed: Olive error')
    );
    consoleSpy.mockRestore();
    container.unmount();
  });

  it('should handle MLIR generation failure gracefully', async () => {
    const container = new RHSContainer();
    container.mount(document.body);
    const dropdown = (container as object).dropdown;
    dropdown.triggerChange('mlir');

    const { lowerONNXToMHLO } =
      await import('@onnx9000/iree-compiler/dist/passes/lower_onnx_to_mhlo.js');
    (lowerONNXToMHLO as object).mockImplementationOnce(() => {
      throw new Error('MLIR error');
    });

    const consoleSpy = vi.spyOn(console, 'error');
    globalEventBus.emit('ONNX_BINARY_GENERATED', new Uint8Array([1, 2, 3]));
    await new Promise((r) => setTimeout(r, 10));
    expect(consoleSpy).toHaveBeenCalledWith(
      expect.stringContaining('MLIR generation failed: MLIR error')
    );
    consoleSpy.mockRestore();
    container.unmount();
  });

  it('should handle Simplification failure gracefully', async () => {
    const container = new RHSContainer();
    container.mount(document.body);
    const dropdown = (container as object).dropdown;
    dropdown.triggerChange('onnx-simplifier');

    const { simplify } = await import('@onnx9000/optimum');
    (simplify as object).mockRejectedValueOnce(new Error('Simplifier error'));

    const consoleSpy = vi.spyOn(console, 'error');
    globalEventBus.emit('ONNX_BINARY_GENERATED', new Uint8Array([1, 2, 3]));
    await new Promise((r) => setTimeout(r, 10));
    expect(consoleSpy).toHaveBeenCalledWith(
      expect.stringContaining('Simplification failed: Simplifier error')
    );
    consoleSpy.mockRestore();
    container.unmount();
  });

  it('should handle generic conversion failure gracefully', async () => {
    const container = new RHSContainer();
    container.mount(document.body);
    const dropdown = (container as object).dropdown;
    dropdown.triggerChange('caffe');

    const { convert } = await import('@onnx9000/converters');
    (convert as object).mockRejectedValueOnce(new Error('Caffe conversion error'));

    const consoleSpy = vi.spyOn(console, 'error');
    globalEventBus.emit('ONNX_BINARY_GENERATED', new Uint8Array([1, 2, 3]));
    await new Promise((r) => setTimeout(r, 10));
    expect(consoleSpy).toHaveBeenCalledWith(
      expect.stringContaining('caffe conversion failed: Caffe conversion error')
    );
    consoleSpy.mockRestore();
  });
});

it('should ignore missing target on change', () => {
  const container = new RHSContainer();
  const dropdown = (container as object).dropdown;
  const tree = (container as object).tree;

  dropdown.triggerChange('fake-framework-does-not-exist');
  expect(tree.updateData).not.toHaveBeenCalledWith(undefined);
});

it('should ignore selection with missing content gracefully', () => {
  const container = new RHSContainer();
  container.mount(document.body);
  const tree = (container as object).tree;
  const editor = (container as object).editor;

  tree.triggerSelect('/fake/output.onnx');
  expect(editor.openFile).toHaveBeenCalledWith(
    '/fake/output.onnx',
    '// Binary representation of /fake/output.onnx',
    'plaintext'
  );
});
