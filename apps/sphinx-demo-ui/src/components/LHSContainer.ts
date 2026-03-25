import { convert } from '@onnx9000/converters';
import { globalEventBus } from '../core/EventBus';
import { Component } from '../core/Component';
import { Dropdown } from './Dropdown';
import { FileTree } from './FileTree';
import { Editor } from './Editor';
import { SplitPane } from './SplitPane';
import { LHS_FRAMEWORKS, LHS_EXAMPLES } from '../data/MockData';
import { Debouncer } from '../core/Debouncer';
import { t } from '../core/I18n';
import { keras2onnx } from '@onnx9000/converters';
import { BufferReader, parseModelProto, serializeModelProto } from '@onnx9000/core';
import { VizGraph, VizNode } from '../core/OnnxAdapter';
import { KerasPythonParser } from '../core/KerasPythonParser';

export class LHSContainer extends Component<HTMLDivElement> {
  private frameworkDropdown!: Dropdown;
  private exampleDropdown!: Dropdown;
  private tree!: FileTree;
  private editor!: Editor;
  private splitPane!: SplitPane;
  private debouncer = new Debouncer();
  private runBtn!: HTMLButtonElement;
  private _boundLanguageChanged: () => void;

  constructor() {
    super();
    this._boundLanguageChanged = () => this.updateI18n();
    this.element = this.render();
  }

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
        } catch (e) {}
      }
    });

    this.exampleDropdown = new Dropdown({
      items: [],
      placeholder: 'Select Example...',
      onChange: (val) => {
        const fw = this.frameworkDropdown.getValue() || 'keras';
        this.selectExample(fw, val);
        try {
          localStorage.setItem(`onnx9000-demo-lhs-example-${fw}`, val);
        } catch (e) {}
      }
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

      const sourceId = this.frameworkDropdown.getValue() || 'keras';

      try {
        let vizGraph: VizGraph;
        let onnxBytes: Uint8Array;

        if (sourceId === 'keras') {
          console.log('[stdout] Validating Keras source...');
          const sourceCode = this.editor.getValue();

          let parsed;
          try {
            if (sourceCode.includes('import keras') || sourceCode.includes('models.Sequential')) {
              console.log('[stdout] Python Keras script detected. Parsing dynamically...');
              parsed = await KerasPythonParser.parse(sourceCode);
            } else {
              parsed = JSON.parse(sourceCode);
            }
          } catch (err: any) {
            console.warn(
              `[stderr] Failed to parse editor content: ${err.message}. Falling back to default mnist model...`
            );
            parsed = JSON.parse(LHS_EXAMPLES['keras'][0].root.children![1].content!);
          }

          const modelJsonString = JSON.stringify(parsed);

          console.log('[stdout] Using @onnx9000/converters keras2onnx...');
          onnxBytes = await keras2onnx(modelJsonString);
          if (onnxBytes.byteLength === 0) {
            const nodes: VizNode[] = [];
            const layers = parsed?.modelTopology?.config?.layers || [];
            let lastOutput = '';

            layers.forEach((layer: any, idx: number) => {
              const name = layer.config?.name || layer.class_name + '_' + idx;
              const input = lastOutput ? [lastOutput] : [];
              const output = [name + ':0'];

              nodes.push({
                id: name,
                name: name,
                opType: layer.class_name,
                inputs: layer.class_name === 'InputLayer' ? [] : input,
                outputs: output,
                attributes: {}
              });
              lastOutput = output[0];
            });

            vizGraph = {
              nodes,
              inputs:
                layers.length > 0
                  ? [{ name: layers[0].config?.name || 'input_1', type: 'float32' }]
                  : [],
              outputs: layers.length > 0 ? [{ name: lastOutput, type: 'float32' }] : []
            };
          } else {
            console.log(
              `[stdout] Conversion successful. Generated ${onnxBytes.byteLength} bytes of ONNX protobuf.`
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
                attributes: {}
              })),
              inputs: graph.inputs.map((i) => ({ name: i.name, type: i.dtype })),
              outputs: graph.outputs.map((o) => ({ name: o.name, type: o.dtype }))
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
          const graph = await convert(sourceId as any, 'onnx', [file]);

          vizGraph = {
            nodes: graph.nodes.map((n: any) => ({
              id: n.name || Math.random().toString(),
              name: n.name || '',
              opType: n.opType,
              inputs: n.inputs,
              outputs: n.outputs,
              attributes: {}
            })),
            inputs: graph.inputs.map((i: any) => ({ name: i.name, type: i.dtype })),
            outputs: graph.outputs.map((o: any) => ({ name: o.name, type: o.dtype }))
          };

          onnxBytes = serializeModelProto(graph as any);
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
        const sourceLabel = LHS_FRAMEWORKS.find((f) => f.value === sourceId)?.label || sourceId;
        globalEventBus.emit('PIPELINE_STEP_ADDED', {
          id: Date.now().toString(),
          description: `${sourceLabel} -> ONNX`,
          state: { sourceFramework: sourceId, targetFramework: targetId, activeFile: '' }
        });
      } catch (err: any) {
        console.error(`[stderr] Error: ${err.stack || err.message || err}`);
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
      storageKey: 'onnx9000-demo-lhs-split'
    });
    this.splitPane.mount(splitArea);

    const { pane1, pane2 } = this.splitPane.getPanes();

    // 3. Tree (Top)
    this.tree = new FileTree({
      root: { name: 'loading...', type: 'directory', path: '/' },
      onSelect: (path) => {
        const node = this.tree['findNode'](this.tree['options'].root, path);
        const content =
          node && node.content
            ? node.content
            : '# ' + path + '\n\n// Example generated by ONNX9000 Demo\n';
        this.editor.openFile(path, content, 'python');
      }
    });
    this.tree.mount(pane1);

    // 4. Editor (Bottom)
    this.editor = new Editor({
      language: 'python',
      initialValue: '# Select a file...',
      onChange: this.debouncer.debounce((val: string) => {
        console.log('Debounced editor change:', val.length, 'bytes');
        // Trigger WASM conversion in the future
      }, 500)
    });
    this.editor.mount(pane2);

    container.appendChild(splitArea);
    return container;
  }

  private updateI18n() {
    if (this.runBtn) {
      if (!this.runBtn.disabled) {
        this.runBtn.textContent = t('lhs.run');
      } else {
        this.runBtn.textContent = t('lhs.converting');
      }
    }
  }

  protected onMount(): void {
    const fw = this.frameworkDropdown.getValue() || 'keras';
    this.updateExampleDropdown(fw);
    globalEventBus.on('LANGUAGE_CHANGED', this._boundLanguageChanged);
    this.onCleanup(() => {
      globalEventBus.off('LANGUAGE_CHANGED', this._boundLanguageChanged);
    });
  }

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

  private selectExample(framework: string, exampleId: string): void {
    const examples = LHS_EXAMPLES[framework];
    if (examples) {
      const ex = examples.find((e) => e.id === exampleId);
      if (ex) {
        this.tree.updateData(ex.root);
        if (ex.initialFile) {
          // Open initial file via mock logic
          const node = this.tree['findNode'](ex.root, ex.initialFile);
          const content =
            node && node.content
              ? node.content
              : '# ' + ex.initialFile + '\n\n// Example generated by ONNX9000 Demo\n';
          this.editor.openFile(ex.initialFile, content, 'python');
        }
      }
    }
  }
}
