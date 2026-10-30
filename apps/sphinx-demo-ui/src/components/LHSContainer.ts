/**
 * @fileoverview LHSContainer.ts
 * Provides LHSContainer functionality for the Sphinx Demo UI.
 */
import { convert, keras2onnx, type SourceFramework } from '@onnx9000/converters';
import {
  BufferReader,
  type Graph,
  type Node,
  parseModelProto,
  serializeModelProto,
  type ValueInfo,
} from '@onnx9000/core';
import { Component } from '../core/Component';
import { Debouncer } from '../core/Debouncer';
import { globalEventBus } from '../core/EventBus';
import { t } from '../core/I18n';
import { KerasPythonParser } from '../core/KerasPythonParser';
import type { VizGraph, VizNode } from '../core/OnnxAdapter';
import { LHS_EXAMPLES, LHS_FRAMEWORKS } from '../data/MockData';
import { Dropdown } from './Dropdown';
import { Editor } from './Editor';
import { type FileNode, FileTree } from './FileTree';
import { SplitPane } from './SplitPane';

/** Represents the `LHSContainer` class. */
export class LHSContainer extends Component<HTMLDivElement> {
  /** Property `frameworkDropdown` of `LHSContainer`. */
  private frameworkDropdown!: Dropdown;
  /** Property `exampleDropdown` of `LHSContainer`. */
  private exampleDropdown!: Dropdown;
  /** Property `tree` of `LHSContainer`. */
  private tree!: FileTree;
  /** Property `editor` of `LHSContainer`. */
  private editor!: Editor;
  /** Property `splitPane` of `LHSContainer`. */
  private splitPane!: SplitPane;
  /** Property `debouncer` of `LHSContainer`. */
  private debouncer = new Debouncer();
  /** Property `runBtn` of `LHSContainer`. */
  private runBtn!: HTMLButtonElement;
  /** Property `_boundLanguageChanged` of `LHSContainer`. */
  private _boundLanguageChanged: () => void;

  /** Initializes a new `LHSContainer` instance. */
  constructor() {
    super();
    this._boundLanguageChanged = () => this.updateI18n();
    this.element = this.render();
  }

  /**
   * Executes `render` on `LHSContainer`.
   * @returns Result of `render`.
   */
  protected render(): HTMLDivElement {
    const container = document.createElement('div');
    container.className = 'demo-pane-lhs';
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

    const dropdownsRow = document.createElement('div');
    dropdownsRow.style.display = 'flex';
    dropdownsRow.style.gap = '8px';

    const defaultFw =
      document.getElementById('interactive-demo-container')?.getAttribute('data-initial-source') ||
      localStorage.getItem('onnx9000-demo-lhs-framework') ||
      'keras';

    this.frameworkDropdown = new Dropdown({
      items: LHS_FRAMEWORKS,
      placeholder: 'Select Source Framework...',
      initialValue: defaultFw,
      onChange: (val) => {
        this.updateExampleDropdown(val);
        try {
          localStorage.setItem('onnx9000-demo-lhs-framework', val);
        } catch {
          // Ignore storage errors
        }
      },
    });

    this.exampleDropdown = new Dropdown({
      items: [],
      placeholder: 'Select Example...',
      onChange: (val) => {
        const fw = this.frameworkDropdown.getValue() as string;
        this.selectExample(fw, val);
        try {
          localStorage.setItem(`onnx9000-demo-lhs-example-${fw}`, val);
        } catch {
          // Ignore storage errors
        }
      },
    });

    this.frameworkDropdown.mount(dropdownsRow);
    this.exampleDropdown.mount(dropdownsRow);
    header.appendChild(dropdownsRow);

    this.runBtn = document.createElement('button');
    this.runBtn.className = 'demo-btn-run-conversion';
    this.runBtn.textContent = t('lhs.run');
    this.runBtn.style.flexShrink = '0';
    this.runBtn.style.width = '100%';
    this.runBtn.onclick = async () => {
      globalEventBus.emit('CONVERSION_STARTED', null);
      console.log('Starting ONNX conversion...');

      this.runBtn.disabled = true;
      this.runBtn.textContent = t('lhs.converting');

      const sourceId = this.frameworkDropdown.getValue() as string;

      try {
        let vizGraph: VizGraph;
        let onnxBytes: Uint8Array;

        if (sourceId === 'keras') {
          console.log('[stdout] Validating Keras source...');
          const sourceCode = this.editor.getValue();

          interface KerasLayerSpec {
            class_name: string;
            config?: { name?: string };
          }
          interface KerasModelParsed {
            modelTopology?: {
              config?: {
                layers?: KerasLayerSpec[];
              };
            };
          }

          let parsed: KerasModelParsed;
          try {
            if (sourceCode.includes('import keras') || sourceCode.includes('models.Sequential')) {
              console.log('[stdout] Python Keras script detected. Parsing dynamically...');
              parsed = (await KerasPythonParser.parse(sourceCode)) as KerasModelParsed;
            } else {
              parsed = JSON.parse(sourceCode) as KerasModelParsed;
            }
          } catch (err) {
            const errMsg = err instanceof Error ? err.message : String(err);
            console.warn(
              `[stderr] Failed to parse editor content: ${errMsg}. Falling back to default mnist model...`,
            );
            parsed = JSON.parse(
              (LHS_EXAMPLES.keras[0].root.children as FileNode[])[1].content as string,
            ) as KerasModelParsed;
          }

          const modelJsonString = JSON.stringify(parsed);

          console.log('[stdout] Using @onnx9000/converters keras2onnx...');
          onnxBytes = await keras2onnx(modelJsonString);
          if (onnxBytes.byteLength === 0) {
            const nodes: VizNode[] = [];
            const layers = parsed?.modelTopology?.config?.layers || [];
            let lastOutput = '';

            layers.forEach((layer: KerasLayerSpec, idx: number) => {
              const name = layer.config?.name || `${layer.class_name}_${idx}`;
              const input = lastOutput ? [lastOutput] : [];
              const output = [`${name}:0`];

              nodes.push({
                id: name,
                name: name,
                opType: layer.class_name,
                inputs: layer.class_name === 'InputLayer' ? [] : input,
                outputs: output,
                attributes: {},
              });
              lastOutput = output[0];
            });

            vizGraph = {
              nodes,
              inputs:
                layers.length > 0
                  ? [{ name: layers[0].config?.name || 'input_1', type: 'float32' }]
                  : [],
              outputs: layers.length > 0 ? [{ name: lastOutput, type: 'float32' }] : [],
            };
          } else {
            console.log(
              `[stdout] Conversion successful. Generated ${onnxBytes.byteLength} bytes of ONNX protobuf.`,
            );
            console.log('[stdout] Parsing ONNX for visualization...');
            const reader = new BufferReader(onnxBytes);
            const graph = await parseModelProto(reader);
            console.log(`[stdout] Graph parsing complete. Nodes: ${graph.nodes.length}`);

            vizGraph = {
              nodes: graph.nodes.map((n) => ({
                id: n.name || Math.random().toString(),
                name: n.name || '',
                opType: n.opType,
                inputs: n.inputs,
                outputs: n.outputs,
                attributes: {},
              })),
              inputs: graph.inputs.map((i) => ({ name: i.name, type: i.dtype })),
              outputs: graph.outputs.map((o) => ({ name: o.name, type: o.dtype })),
            };
          }
        } else {
          console.log(`[stdout] Converting ${sourceId} to ONNX using @onnx9000/converters...`);

          const fileContent = this.editor.getValue();
          let ext = '.txt';
          if (sourceId === 'caffe') ext = '.prototxt';
          else if (sourceId === 'mxnet') ext = '.json';
          else if (sourceId === 'tensorflow') ext = '.pbtxt';
          else if (sourceId === 'paddle') ext = '.json';
          else if (sourceId === 'onnxscript') ext = '.py';
          else if (sourceId === 'scikitlearn') ext = '.json';
          else if (sourceId === 'xgboost') ext = '.json';
          else if (sourceId === 'catboost') ext = '.json';
          else if (sourceId === 'sparkml') ext = '.json';

          let filename = `model_file${ext}`;
          if (sourceId === 'paddle') filename = '__model__';
          const file = new File([fileContent], filename, { type: 'text/plain' });
          const graph = (await convert(sourceId as SourceFramework, 'onnx', [file])) as Graph;

          vizGraph = {
            nodes: graph.nodes.map((n: Node) => ({
              id: n.name || Math.random().toString(),
              name: n.name || '',
              opType: n.opType,
              inputs: n.inputs,
              outputs: n.outputs,
              attributes: {},
            })),
            inputs: graph.inputs.map((i: ValueInfo) => ({ name: i.name, type: i.dtype })),
            outputs: graph.outputs.map((o: ValueInfo) => ({ name: o.name, type: o.dtype })),
          };

          onnxBytes = serializeModelProto(graph);
        }

        globalEventBus.emit('ONNX_GRAPH_GENERATED', vizGraph);
        globalEventBus.emit('ONNX_BINARY_GENERATED', onnxBytes);
        console.log('Conversion completed successfully.');
        globalEventBus.emit('CONVERSION_COMPLETED', null);

        // Add to pipeline
        const targetId =
          document
            .getElementById('interactive-demo-container')
            ?.getAttribute('data-initial-target') ||
          localStorage.getItem('onnx9000-demo-rhs-target') ||
          'onnx';
        const sourceLabel = (
          LHS_FRAMEWORKS.find((f) => f.value === sourceId) as (typeof LHS_FRAMEWORKS)[number]
        ).label;
        globalEventBus.emit('PIPELINE_STEP_ADDED', {
          id: Date.now().toString(),
          description: `${sourceLabel} -> ONNX`,
          state: { sourceFramework: sourceId, targetFramework: targetId, activeFile: '' },
        });
      } catch (err) {
        const detail = err instanceof Error ? err.stack || err.message : String(err);
        console.error(`[stderr] Error: ${detail}`);
        console.error('[stderr] Failed to generate ONNX representation.');
        console.error('Conversion unsuccessful.');
      } finally {
        this.runBtn.disabled = false;
        this.runBtn.textContent = t('lhs.run');
      }
    };
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
      storageKey: 'onnx9000-demo-lhs-split',
    });
    this.splitPane.mount(splitArea);

    const { pane1, pane2 } = this.splitPane.getPanes();

    // 3. Tree (Top)
    this.tree = new FileTree({
      root: { name: 'loading...', type: 'directory', path: '/' },
      onSelect: (path) => {
        const node = this.tree.findNode(this.tree.options.root, path);
        const content = node?.content
          ? node.content
          : `# ${path}\n\n// Example generated by ONNX9000 Demo\n`;
        this.editor.openFile(path, content, 'python');
      },
    });
    this.tree.mount(pane1);

    // 4. Editor (Bottom)
    this.editor = new Editor({
      language: 'python',
      initialValue: '# Select a file...',
      onChange: this.debouncer.debounce((val: string) => {
        console.log('Debounced editor change:', val.length, 'bytes');
        // Trigger WASM conversion in the future
      }, 500),
    });
    this.editor.mount(pane2);

    container.appendChild(splitArea);
    return container;
  }

  /**
   * Executes `updateI18n` on `LHSContainer`.
   * @returns Result of `updateI18n`.
   */
  private updateI18n() {
    if (this.runBtn) {
      if (!this.runBtn.disabled) {
        this.runBtn.textContent = t('lhs.run');
      } else {
        this.runBtn.textContent = t('lhs.converting');
      }
    }
  }

  /** Executes `onMount` on `LHSContainer`. */
  protected onMount(): void {
    const fw = this.frameworkDropdown.getValue() as string;
    this.updateExampleDropdown(fw);
    globalEventBus.on('LANGUAGE_CHANGED', this._boundLanguageChanged);
    this.onCleanup(() => {
      globalEventBus.off('LANGUAGE_CHANGED', this._boundLanguageChanged);
    });
  }

  /**
   * Executes `updateExampleDropdown` on `LHSContainer`.
   * @param framework Parameter `framework`.
   */
  private updateExampleDropdown(framework: string): void {
    const examples = LHS_EXAMPLES[framework];
    if (examples && examples.length > 0) {
      this.exampleDropdown.updateItems(examples.map((ex) => ({ value: ex.id, label: ex.label })));

      let defaultEx = localStorage.getItem(`onnx9000-demo-lhs-example-${framework}`);
      if (!defaultEx || !examples.find((ex) => ex.id === defaultEx)) {
        defaultEx = examples[0].id;
      }

      this.exampleDropdown.select(defaultEx);
      this.selectExample(framework, defaultEx);
    } else {
      this.exampleDropdown.updateItems([]);
    }
  }

  /**
   * Executes `selectExample` on `LHSContainer`.
   * @param framework Parameter `framework`.
   * @param exampleId Parameter `exampleId`.
   */
  private selectExample(framework: string, exampleId: string): void {
    const examples = LHS_EXAMPLES[framework];
    if (examples) {
      const ex = examples.find((e) => e.id === exampleId);
      if (ex) {
        this.tree.updateData(ex.root);
        if (ex.initialFile) {
          // Open initial file via mock logic
          const node = this.tree.findNode(ex.root, ex.initialFile);
          const content = node?.content
            ? node.content
            : `# ${ex.initialFile}\n\n// Example generated by ONNX9000 Demo\n`;
          this.editor.openFile(ex.initialFile, content, 'python');
        }
      }
    }
  }
}
