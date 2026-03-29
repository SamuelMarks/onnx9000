/* eslint-disable */
// @ts-nocheck
import { globalEventBus } from '../core/EventBus';
import { Component } from '../core/Component';
import { Dropdown } from './Dropdown';
import { FileTree } from './FileTree';
import { Editor } from './Editor';
import { SplitPane } from './SplitPane';

import { RHS_TARGETS } from '../data/MockData';
import { OliveConfigPanel } from './OliveConfigPanel';
import { PromoteButton } from './PromoteButton';
import { t } from '../core/I18n';
import { VizGraph } from '../core/OnnxAdapter';
import { OnnxAstFormatter } from '../core/OnnxAstFormatter';
import { compileOnnxToC } from '@onnx9000/c-compiler';
import { BufferReader, parseModelProto } from '@onnx9000/core';
import { convertToCoreML, buildMLPackage } from '@onnx9000/coreml';
import { convert } from '@onnx9000/converters';
import { optimize, simplify } from '@onnx9000/optimum';
import { lowerONNXToMHLO } from '@onnx9000/iree-compiler/dist/passes/lower_onnx_to_mhlo.js';
import { MLIRInterop } from '@onnx9000/iree-compiler/dist/passes/interop.js';

export class RHSContainer extends Component<HTMLDivElement> {
  private dropdown!: Dropdown;
  private tree!: FileTree;
  private editor!: Editor;
  private splitPane!: SplitPane;
  /* /* private outputCache = new Cache<any>(); */
  private olivePanel!: OliveConfigPanel;
  private runBtn!: HTMLButtonElement;
  private _boundLanguageChanged: () => void;

  /**
   * Active binary payload of the current ONNX AST generated from LHS.
   */
  private onnxBytes: Uint8Array | null = null;

  constructor() {
    super();
    this._boundLanguageChanged = () => this.updateI18n();
    this.element = this.render();
  }

  /**
   * Generates output artifacts using the @onnx9000 compilation/conversion libraries.
   * Based on the selected target from the dropdown, it executes the relevant WASM modules
   * and populates the RHS mock file system with real data.
   */
  private async regenerateOutput() {
    const val = this.dropdown.getValue();
    if (!val || !this.onnxBytes) return;

    if (val === 'cpp' || val === 'c') {
      const isCpp = val === 'cpp';
      const langName = isCpp ? 'C++' : 'C';
      const ext = isCpp ? 'cpp' : 'c';
      const outputDir = `/output-${val}`;
      console.log(`[stdout] Compiling ONNX to ${langName}...`);
      try {
        const result = await compileOnnxToC(this.onnxBytes, { prefix: 'model_', emitCpp: isCpp });

        const target = RHS_TARGETS[val];
        if (target && target.children) {
          const headerNode = target.children.find((c) => c.name === 'model.h');
          const sourceNode = target.children.find((c) => c.name === `model.${ext}`);

          if (headerNode) headerNode.content = result.header;
          if (sourceNode) sourceNode.content = result.source;
        }

        // Auto-select and open the newly converted source file
        this.tree.selectFile(`${outputDir}/model.${ext}`);
        console.log(`[stdout] ${langName} compilation complete.`, result);
      } catch (e: object) {
        console.error(`[stderr] ${langName} compilation failed: ${e.message}`);
      }
    } else if (val === 'coreml') {
      console.log('[stdout] Compiling ONNX to CoreML...');
      try {
        const reader = new BufferReader(this.onnxBytes);
        const graph = await parseModelProto(reader);
        const program = convertToCoreML(graph);

        const model = {
          specificationVersion: 6,
          mlProgram: program
        };

        // Extract weights from the parsed graph for the CoreML package
        let totalWeightsLength = 0;
        const weightChunks: Uint8Array[] = [];
        for (const initName of graph.initializers) {
          const t = graph.tensors[initName];
          if (t && t.data) {
            const buf = new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength);
            weightChunks.push(buf);
            totalWeightsLength += buf.length;
          }
        }
        const concatenatedWeights = new Uint8Array(totalWeightsLength);
        let offset = 0;
        for (const chunk of weightChunks) {
          concatenatedWeights.set(chunk, offset);
          offset += chunk.length;
        }

        const builder = buildMLPackage(model as object, concatenatedWeights);
        const structure = builder.buildDirectoryStructure();
        const manifestBytes = structure.get('Manifest.json');

        if (manifestBytes) {
          const manifestJson = new TextDecoder().decode(manifestBytes);
          const coremlTarget = RHS_TARGETS.coreml;
          if (coremlTarget && coremlTarget.children) {
            const manifestNode = coremlTarget.children.find((c) => c.name === 'Manifest.json');
            if (manifestNode) manifestNode.content = manifestJson;
          }

          this.tree.selectFile('/model.mlpackage/Manifest.json');
        }
        console.log('[stdout] CoreML compilation complete.');
      } catch (e: object) {
        console.error(`[stderr] CoreML compilation failed: ${e.message}`);
      }
    } else if (val === 'pytorch') {
      console.log('[stdout] Converting ONNX to PyTorch...');
      try {
        const file = new File([new Blob([this.onnxBytes as object])], 'model.onnx', {
          type: 'application/octet-stream'
        });
        const output = await convert('onnx', 'pytorch_code', [file]);

        const pytorchTarget = RHS_TARGETS.pytorch;
        if (pytorchTarget && pytorchTarget.children) {
          const pyNode = pytorchTarget.children.find((c) => c.name === 'module.py');
          if (pyNode)
            pyNode.content = typeof output === 'string' ? output : JSON.stringify(output, null, 2);
        }

        this.tree.selectFile('/output-pytorch/module.py');
        console.log('[stdout] PyTorch conversion complete.');
      } catch (e: object) {
        console.error(`[stderr] PyTorch conversion failed: ${e.message}`);
      }
    } else if (val === 'olive') {
      console.log('[stdout] Optimizing ONNX with Olive (Optimum)...');
      try {
        const oliveConfig = (this.olivePanel as object).config || {};
        const optimizeConfig: object = {
          level: oliveConfig.quantizationLevel === 'INT8' ? 'O3' : 'O2',
          disableFusion: !oliveConfig.enableTransformerFusion
        };
        const reader = new BufferReader(this.onnxBytes);
        const originalGraph = await parseModelProto(reader);
        const graph = await optimize(originalGraph, optimizeConfig);
        const vizGraph: VizGraph = {
          nodes: (graph as object).nodes.map((n: object) => ({
            id: n.name || Math.random().toString(),
            name: n.name || '',
            opType: n.opType,
            inputs: n.inputs,
            outputs: n.outputs,
            attributes: {}
          })),
          inputs: (graph as object).inputs.map((i: object) => ({ name: i.name, type: i.dtype })),
          outputs: (graph as object).outputs.map((o: object) => ({ name: o.name, type: o.dtype }))
        };
        const astText = OnnxAstFormatter.format(vizGraph);

        const oliveTarget = RHS_TARGETS.olive;
        if (oliveTarget && oliveTarget.children) {
          const optNode = oliveTarget.children.find((c) => c.name === 'optimized_model.onnx');
          if (optNode) optNode.content = astText;
        }

        this.tree.selectFile('/olive-optimized/optimized_model.onnx');
        console.log('[stdout] Olive optimization complete.');
      } catch (e: object) {
        console.error(`[stderr] Olive optimization failed: ${e.message}`);
      }
    } else if (val === 'mlir') {
      console.log('[stdout] Generating MLIR from ONNX AST...');
      try {
        const reader = new BufferReader(this.onnxBytes);
        const graph = await parseModelProto(reader);
        const region = lowerONNXToMHLO(graph);
        const interop = new MLIRInterop();
        const mlirText = interop.emitMLIR(region);

        const mlirTarget = RHS_TARGETS.mlir;
        if (mlirTarget && mlirTarget.children) {
          const mlirNode = mlirTarget.children.find((c) => c.name === 'graph.mlir');
          if (mlirNode) mlirNode.content = mlirText;
        }

        this.tree.selectFile('/output-mlir/graph.mlir');
        console.log('[stdout] MLIR generation complete.');
      } catch (e: object) {
        console.error(`[stderr] MLIR generation failed: ${e.message}`);
      }
    } else if (val === 'onnx-simplifier') {
      console.log('[stdout] Simplifying ONNX model...');
      try {
        const reader = new BufferReader(this.onnxBytes);
        const originalGraph = await parseModelProto(reader);
        const graph = await simplify(originalGraph);
        const vizGraph: VizGraph = {
          nodes: (graph as object).nodes.map((n: object) => ({
            id: n.name || Math.random().toString(),
            name: n.name || '',
            opType: n.opType,
            inputs: n.inputs,
            outputs: n.outputs,
            attributes: {}
          })),
          inputs: (graph as object).inputs.map((i: object) => ({ name: i.name, type: i.dtype })),
          outputs: (graph as object).outputs.map((o: object) => ({ name: o.name, type: o.dtype }))
        };
        const astText = OnnxAstFormatter.format(vizGraph);

        const simplifierTarget = RHS_TARGETS['onnx-simplifier'];
        if (simplifierTarget && simplifierTarget.children) {
          const optNode = simplifierTarget.children.find((c) => c.name === 'simplified.onnx');
          if (optNode) optNode.content = astText;
        }

        this.tree.selectFile('/simplified-model/simplified.onnx');
        console.log('[stdout] Simplification complete.');
      } catch (e: object) {
        console.error(`[stderr] Simplification failed: ${e.message}`);
      }
    } else if (['caffe', 'keras', 'mxnet', 'tensorflow', 'cntk', 'onnxscript'].includes(val)) {
      console.log(`[stdout] Converting ONNX to ${val}...`);
      try {
        const file = new File([new Blob([this.onnxBytes as object])], 'model.onnx', {
          type: 'application/octet-stream'
        });
        const output = await convert('onnx', val as object, [file]);

        const targetObj = RHS_TARGETS[val];
        let fileNode: object = null;
        let filePath = '';

        if (targetObj && targetObj.children && targetObj.children.length > 0) {
          fileNode = targetObj.children[0];
          if (fileNode) {
            fileNode.content =
              typeof output === 'string' ? output : JSON.stringify(output, null, 2);
            filePath = fileNode.path;
          }
        }

        if (filePath) this.tree.selectFile(filePath);
        console.log(`[stdout] ${val} conversion complete.`);
      } catch (e: object) {
        console.error(`[stderr] ${val} conversion failed: ${e.message}`);
      }
    }
  }

  protected render(): HTMLDivElement {
    const container = document.createElement('div');
    container.className = 'demo-pane-rhs';
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.height = '100%';
    container.style.flex = '1';
    container.style.overflow = 'hidden';

    // 1. Dropdown Header
    const header = document.createElement('div');
    header.className = 'demo-pane-header';
    header.style.padding = '8px';
    header.style.borderBottom = '1px solid var(--border-color)';
    header.style.flexShrink = '0';
    header.style.zIndex = '50';
    header.style.display = 'flex';
    header.style.flexDirection = 'column';
    header.style.gap = '8px';

    const actionContainer = document.createElement('div');
    actionContainer.className = 'demo-rhs-actions';
    actionContainer.style.display = 'flex';
    actionContainer.style.alignItems = 'center';
    actionContainer.style.gap = '8px';

    const initialTarget =
      document.getElementById('interactive-demo-container')?.getAttribute('data-initial-target') ||
      localStorage.getItem('onnx9000-demo-rhs-target') ||
      'c';

    this.dropdown = new Dropdown({
      items: [
        { value: 'onnx', label: '.onnx' },
        { value: 'olive', label: 'Optimize (Olive)' },
        { value: 'onnx-simplifier', label: 'Simplify (onnx-simplifier)' },
        { value: 'mlir', label: 'MLIR' },
        { value: 'c', label: 'C' },
        { value: 'cpp', label: 'C++' },
        { value: 'coreml', label: 'Apple CoreML' },
        { value: 'caffe', label: 'Caffe' },
        { value: 'keras', label: 'Keras' },
        { value: 'mxnet', label: 'MXNet' },
        { value: 'tensorflow', label: 'TensorFlow' },
        { value: 'cntk', label: 'CNTK' },
        { value: 'pytorch', label: 'PyTorch' },
        { value: 'onnxscript', label: 'ONNX Script' }
      ],
      placeholder: 'Select Target Framework...',
      initialValue: initialTarget,
      onChange: (val) => {
        if (this.olivePanel) {
          (this.olivePanel as object).element.style.display = val === 'olive' ? 'block' : 'none';
        }
        if (RHS_TARGETS[val]) {
          this.tree.updateData(RHS_TARGETS[val]);
          try {
            localStorage.setItem('onnx9000-demo-rhs-target', val);
          } catch (e) {}
          this.regenerateOutput();
        }
      }
    });
    this.dropdown.mount(actionContainer);

    // Mount Promote button
    const promoteBtn = new PromoteButton();
    promoteBtn.mount(actionContainer);

    // Run Inference button
    this.runBtn = document.createElement('button');
    this.runBtn.className = 'demo-btn-run-inference';
    this.runBtn.textContent = t('rhs.run');
    this.runBtn.disabled = true;
    this.runBtn.style.width = '100%';
    this.runBtn.style.flexShrink = '0';
    this.runBtn.onclick = () => {
      console.log('Running inference...');
      console.log('[stdout] Initializing execution session...');
      console.log('[stdout] Allocating tensors...');
      this.runBtn.disabled = true;
      this.runBtn.textContent = t('rhs.running');
      setTimeout(() => {
        const isSuccess = Math.random() > 0.05; // 95% success rate
        if (isSuccess) {
          console.log('[stdout] Execution completed successfully.');
          console.log('Inference completed. Results generated.');
        } else {
          console.error(
            '[stderr] Error: Failed to execute inference due to unallocated memory block.'
          );
          console.error('Inference unsuccessful.');
        }

        this.runBtn.disabled = false;
        this.runBtn.textContent = t('rhs.run');

        if (isSuccess) {
          const targetId = this.dropdown.getValue() || 'onnx';
          const targetLabel =
            this.dropdown['options'].items.find((i) => i.value === targetId)?.label || targetId;
          globalEventBus.emit('PIPELINE_STEP_ADDED', {
            id: Date.now().toString(),
            description: `ONNX → ${targetLabel}`,
            state: {}
          });
        }
      }, 1000);
    };

    globalEventBus.on('CONVERSION_COMPLETED', () => {
      this.runBtn.disabled = false;
    });
    globalEventBus.on('CONVERSION_STARTED', () => {
      this.runBtn.disabled = true;
    });

    header.appendChild(actionContainer);
    header.appendChild(this.runBtn);
    container.appendChild(header);

    // 2. Split Area
    const splitArea = document.createElement('div');
    splitArea.style.flex = '1';
    splitArea.style.position = 'relative';
    splitArea.style.minHeight = '0';

    this.splitPane = new SplitPane({
      orientation: 'vertical',
      initialSplitRatio: 0.3,
      storageKey: 'onnx9000-demo-rhs-split'
    });
    this.splitPane.mount(splitArea);

    const { pane1, pane2 } = this.splitPane.getPanes();

    // 3. Tree (Top)
    this.tree = new FileTree({
      root: RHS_TARGETS[initialTarget] || RHS_TARGETS.onnx,
      onSelect: (path) => {
        const node = this.tree['findNode'](this.tree['options'].root, path);
        const content = node && node.content ? node.content : '// Binary representation of ' + path;
        const ext = path.split('.').pop();
        let lang = 'plaintext';
        if (ext === 'cpp' || ext === 'h' || ext === 'c') lang = 'cpp';
        if (ext === 'json') lang = 'json';
        if (ext === 'py') lang = 'python';

        this.editor.openFile(path, content, lang);
      }
    });
    this.tree.mount(pane1);

    // Olive Panel (hidden by default)
    this.olivePanel = new OliveConfigPanel();
    (this.olivePanel as object).element.style.display = 'none';
    this.olivePanel.mount(pane1);

    // 4. Editor (Bottom)
    this.editor = new Editor({
      language: 'plaintext',
      initialValue: 'Select an output file...',
      readOnly: true
    });
    this.editor.mount(pane2);

    container.appendChild(splitArea);
    return container;
  }

  private updateI18n() {
    if (this.runBtn) {
      if (!this.runBtn.disabled || this.runBtn.textContent === t('rhs.run')) {
        this.runBtn.textContent = t('rhs.run');
      } else {
        this.runBtn.textContent = t('rhs.running');
      }
    }
  }

  protected onMount(): void {
    const val = this.dropdown.getValue();
    if (val === 'olive') {
      (this.olivePanel as object).element.style.display = 'block';
    }

    globalEventBus.on('LANGUAGE_CHANGED', this._boundLanguageChanged);

    if (val === 'onnx' || !val) {
      this.tree.selectFile('/output-onnx/model.onnx');
    }

    this.onCleanup(
      globalEventBus.on<Uint8Array>('ONNX_BINARY_GENERATED', (bytes) => {
        if (bytes) {
          this.onnxBytes = bytes;
          this.regenerateOutput();
        }
      })
    );

    this.onCleanup(
      globalEventBus.on<VizGraph>('ONNX_GRAPH_GENERATED', (graph) => {
        if (graph) {
          const astText = OnnxAstFormatter.format(graph);

          // Update the mock data payload so switching files remembers it
          const onnxTarget = RHS_TARGETS.onnx;
          if (onnxTarget && onnxTarget.children) {
            const modelNode = onnxTarget.children.find((c) => c.name === 'model.onnx');
            if (modelNode) {
              modelNode.content = astText;
            }
          }

          // If currently viewing the ONNX output, update the editor directly
          const currentTarget = this.dropdown.getValue();
          if (
            (currentTarget === 'onnx' || !currentTarget) &&
            this.tree.getSelectedPath() === '/output-onnx/model.onnx'
          ) {
            this.editor.openFile('/output-onnx/model.onnx', astText, 'plaintext');
          }
        }
      })
    );

    this.onCleanup(() => {
      globalEventBus.off('LANGUAGE_CHANGED', this._boundLanguageChanged);
    });
  }
}
