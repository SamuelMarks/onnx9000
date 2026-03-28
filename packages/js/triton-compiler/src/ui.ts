export class TritonCompilerElement extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
  }

  connectedCallback() {
    this.shadowRoot!.innerHTML = `
      <style>
        :host { display: block; padding: 20px; font-family: sans-serif; }
        .editor { font-family: monospace; white-space: pre; background: #2d2d2d; color: #f8f8f2; padding: 10px; flex: 1; min-height: 400px; overflow: auto; border-radius: 4px; }
        .panel { display: flex; gap: 20px; margin-top: 15px; }
        .controls { display: flex; gap: 15px; align-items: center; margin-bottom: 15px; }
        #dropzone { border: 2px dashed #aaa; padding: 40px; text-align: center; border-radius: 8px; cursor: pointer; background: #f9f9f9; }
        #dropzone.hover { border-color: #007bff; background: #e9f5ff; }
        .slider-group { display: flex; flex-direction: column; font-size: 12px; }
        label { font-weight: bold; }
        .graph-view { border: 1px solid #ccc; height: 300px; display: flex; align-items: center; justify-content: center; background: #fafafa; margin-bottom: 15px; }
      </style>
      <div>
        <h2>onnx9000.triton Visual Compiler</h2>
        
        <div id="dropzone">
          <p>132. Drag and drop ONNX model here</p>
        </div>

        <div class="graph-view" id="graph-container">
          <!-- 133. Display the interactive ONNX Graph (via onnx9000.modifier) -->
          <p>Graph Viewer (Shift-click to select nodes)</p>
        </div>

        <div class="controls">
          <button id="gen" style="padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">Generate Triton Kernel</button>
          <button id="save" style="padding: 8px 16px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer;">Save Python</button>
          
          <!-- 139. Support tweaking BLOCK_SIZE preferences visually via sliders -->
          <div class="slider-group">
            <label>BLOCK_M: <span id="bm-val">64</span></label>
            <input type="range" id="blockM" min="16" max="256" step="16" value="64">
          </div>
          <div class="slider-group">
            <label>BLOCK_N: <span id="bn-val">64</span></label>
            <input type="range" id="blockN" min="16" max="256" step="16" value="64">
          </div>
          <div class="slider-group">
            <label>BLOCK_K: <span id="bk-val">64</span></label>
            <input type="range" id="blockK" min="16" max="256" step="16" value="64">
          </div>
        </div>

        <div class="panel">
          <!-- 136. Display generated Python source code in Monaco Editor -->
          <div style="flex: 1; display: flex; flex-direction: column;">
            <h4>Triton Python Source</h4>
            <div id="output" class="editor" contenteditable="true"></div>
          </div>
          <!-- 137. Display generated WGSL source code in adjacent Monaco Editor -->
          <div style="flex: 1; display: flex; flex-direction: column;">
            <h4>WebGPU WGSL Source</h4>
            <div id="wgsl-output" class="editor" contenteditable="true"></div>
          </div>
        </div>
      </div>
    `;

    const dropzone = this.shadowRoot!.querySelector('#dropzone')!;
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('hover');
    });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('hover'));
    dropzone.addEventListener('drop', (e: any) => {
      e.preventDefault();
      dropzone.classList.remove('hover');
      const file = e.dataTransfer.files[0];
      if (file) {
        dropzone.innerHTML = `<p>Loaded: ${file.name}</p>`;
        this.dispatchEvent(new CustomEvent('model-loaded', { detail: { file } }));
      }
    });

    const bindSlider = (id: string, valId: string) => {
      const slider = this.shadowRoot!.querySelector(id) as HTMLInputElement;
      const val = this.shadowRoot!.querySelector(valId)!;
      slider.addEventListener('input', () => {
        val.textContent = slider.value;
      });
    };
    bindSlider('#blockM', '#bm-val');
    bindSlider('#blockN', '#bn-val');
    bindSlider('#blockK', '#bk-val');

    this.shadowRoot!.querySelector('#gen')!.addEventListener('click', () => {
      const bm = (this.shadowRoot!.querySelector('#blockM') as HTMLInputElement).value;
      const bn = (this.shadowRoot!.querySelector('#blockN') as HTMLInputElement).value;
      const bk = (this.shadowRoot!.querySelector('#blockK') as HTMLInputElement).value;
      this.dispatchEvent(
        new CustomEvent('generate-requested', { detail: { blockM: bm, blockN: bn, blockK: bk } }),
      );
    });

    this.shadowRoot!.querySelector('#save')!.addEventListener('click', () => {
      const code = this.shadowRoot!.querySelector('#output')!.textContent;
      const blob = new Blob([code || ''], { type: 'text/x-python' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'fused_kernel.py';
      a.click();
    });
  }

  bundle(graph: any) {
    const uniforms = graph.inputs
      .filter((i: any) => i.shape && i.shape.length === 0)
      .map((i: any) => i.name);
    return `
      export async function run(device, inputs) {
        const shaderModule = device.createShaderModule({ code: \`...\` });
        const pipeline = await device.createComputePipelineAsync({
          layout: 'auto',
          compute: { module: shaderModule, entryPoint: 'main' }
        });
        // Uniforms: ${uniforms.join(', ')}
      }
    `;
  }

  setCode(pythonCode: string, wgslCode: string = '') {
    // 138. Provide realtime syntax highlighting and formatting (simulated via basic text content for now)
    this.shadowRoot!.querySelector('#output')!.textContent = pythonCode;
    this.shadowRoot!.querySelector('#wgsl-output')!.textContent =
      wgslCode || '// WGSL output here...';
  }
}

customElements.define('triton-compiler-ui', TritonCompilerElement);
