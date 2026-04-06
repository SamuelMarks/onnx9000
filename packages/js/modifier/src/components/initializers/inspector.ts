import { Graph, Tensor } from '@onnx9000/core';
import { GraphMutator } from '../../GraphMutator.js';

/**
 * Visual editor for weight tensors and initializers.
 * Supports Phase 6 features of ONNX29 checklist.
 */
export class InitializerInspector {
  container: HTMLElement;
  mutator: GraphMutator;

  constructor(container: HTMLElement, mutator: GraphMutator) {
    this.container = container;
    this.mutator = mutator;
  }

  render(tensor: Tensor) {
    this.container.innerHTML = '';

    if (!tensor.data) {
      this.container.textContent = 'No internal data present (external data or empty).';
      return;
    }

    const arr = this._getTypedArray(tensor);
    if (!arr) {
      this.container.textContent = `Unsupported view for DType: ${tensor.dtype}`;
      return;
    }

    this._renderStats(arr, tensor);
    this._renderSize(arr);
    this._renderHeatmap(arr, tensor.shape);
    this._renderScalarEditor(arr, tensor);
    this._renderActions(arr, tensor);
  }

  private _getTypedArray(
    tensor: Tensor,
  ): Float32Array | Int32Array | Uint8Array | Float64Array | null {
    if (!tensor.data) return null;
    const buf = tensor.data.buffer;
    switch (tensor.dtype) {
      case 'float32':
        return new Float32Array(buf, tensor.data.byteOffset, tensor.data.byteLength / 4);
      case 'float64':
        return new Float64Array(buf, tensor.data.byteOffset, tensor.data.byteLength / 8);
      case 'int32':
        return new Int32Array(buf, tensor.data.byteOffset, tensor.data.byteLength / 4);
      case 'uint8':
        return new Uint8Array(buf, tensor.data.byteOffset, tensor.data.byteLength);
      // For 16-bit we'd need float16 logic, fallback or null here.
      default:
        return null;
    }
  }

  // 76. Inspector: Min, Max, Mean, Variance
  private _renderStats(arr: ReturnType<typeof JSON.parse>, tensor: Tensor) {
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    const len = arr.length;
    for (let i = 0; i < len; i++) {
      const v = arr[i];
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
    }
    const mean = sum / len;

    let sqSum = 0;
    for (let i = 0; i < len; i++) {
      const diff = arr[i] - mean;
      sqSum += diff * diff;
    }
    const variance = len > 1 ? sqSum / (len - 1) : 0;

    const statsDiv = this._createSection('Statistics');
    statsDiv.innerHTML += `
      <div style="font-size: 11px;">
        <div><strong>Min:</strong> ${min.toFixed(4)}</div>
        <div><strong>Max:</strong> ${max.toFixed(4)}</div>
        <div><strong>Mean:</strong> ${mean.toFixed(4)}</div>
        <div><strong>Var:</strong> ${variance.toFixed(4)}</div>
      </div>
    `;
    this.container.appendChild(statsDiv);
  }

  // 84. Track exact byte sizes
  private _renderSize(arr: ReturnType<typeof JSON.parse>) {
    const sizeDiv = this._createSection('Memory Footprint');
    sizeDiv.innerHTML += `
      <div style="font-size: 11px;">
        <strong>Bytes:</strong> ${arr.byteLength.toLocaleString()} B
      </div>
    `;
    this.container.appendChild(sizeDiv);
  }

  // 77. Render small 2D weights as visual pixel grids (heatmaps)
  private _renderHeatmap(arr: ReturnType<typeof JSON.parse>, shape: (number | string)[]) {
    // Only attempt 2D or 4D where last two dims are small
    const numDims = shape.length;
    if (numDims < 2) return;

    const h = Number(shape[numDims - 2]);
    const w = Number(shape[numDims - 1]);

    // Check if it's a small grid (e.g. up to 16x16)
    if (Number.isNaN(h) || Number.isNaN(w) || h > 16 || w > 16 || h < 1 || w < 1) return;

    // Pick first channel / filter
    const gridLen = h * w;
    if (arr.length < gridLen) return;

    const heatmapDiv = this._createSection(`Heatmap (First ${h}x${w} slice)`);
    const canvas = document.createElement('canvas');
    canvas.width = w * 10;
    canvas.height = h * 10;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Normalize
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < gridLen; i++) {
      if (arr[i] < min) min = arr[i];
      if (arr[i] > max) max = arr[i];
    }
    const range = Math.max(max - min, 1e-5);

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const val = arr[y * w + x];
        const norm = (val - min) / range; // 0 to 1
        const color = Math.floor(norm * 255);
        ctx.fillStyle = `rgb(${color}, 100, ${255 - color})`;
        ctx.fillRect(x * 10, y * 10, 10, 10);
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.strokeRect(x * 10, y * 10, 10, 10);
      }
    }
    heatmapDiv.appendChild(canvas);
    this.container.appendChild(heatmapDiv);
  }

  // 78. Support explicitly editing scalar initializer values via text
  private _renderScalarEditor(arr: ReturnType<typeof JSON.parse>, tensor: Tensor) {
    if (arr.length !== 1) return;

    const div = this._createSection('Scalar Editor');
    const input = document.createElement('input');
    input.type = 'number';
    input.value = String(arr[0]);
    input.style.width = '100%';
    input.onchange = (e) => {
      const val = parseFloat((e.target as HTMLInputElement).value);
      if (!Number.isNaN(val)) {
        const newArr = new arr.constructor(1);
        newArr[0] = val;
        this.mutator.updateInitializer(tensor.name, newArr);
      }
    };
    div.appendChild(input);
    this.container.appendChild(div);
  }

  private _renderActions(arr: ReturnType<typeof JSON.parse>, tensor: Tensor) {
    const actDiv = this._createSection('Actions');
    actDiv.style.display = 'flex';
    actDiv.style.flexDirection = 'column';
    actDiv.style.gap = '4px';

    // 79. Zero out an initializer
    const zeroBtn = document.createElement('button');
    zeroBtn.textContent = 'Zero Out';
    zeroBtn.onclick = () => {
      const newArr = new arr.constructor(arr.length); // auto 0-filled
      this.mutator.updateInitializer(tensor.name, newArr);
    };
    actDiv.appendChild(zeroBtn);

    // 80. Random noise injection
    const noiseBtn = document.createElement('button');
    noiseBtn.textContent = 'Add Noise (Fuzz)';
    noiseBtn.onclick = () => {
      const newArr = new arr.constructor(arr.length);
      for (let i = 0; i < arr.length; i++) {
        newArr[i] = arr[i] + (Math.random() - 0.5) * 0.1 * (arr[i] || 1); // small relative noise
      }
      this.mutator.updateInitializer(tensor.name, newArr);
    };
    actDiv.appendChild(noiseBtn);

    // 85. Prune by magnitude
    const pruneBtn = document.createElement('button');
    pruneBtn.textContent = 'Prune (< 1e-3)';
    pruneBtn.onclick = () => {
      const newArr = new arr.constructor(arr.length);
      for (let i = 0; i < arr.length; i++) {
        newArr[i] = Math.abs(arr[i]) < 1e-3 ? 0 : arr[i];
      }
      this.mutator.updateInitializer(tensor.name, newArr);
    };
    actDiv.appendChild(pruneBtn);

    // 83. Precision cast
    const castBtn = document.createElement('button');
    castBtn.textContent = 'Cast to FP32';
    castBtn.onclick = () => {
      if (tensor.dtype !== 'float32') {
        const origType = tensor.dtype;
        const fp32Arr = new Float32Array(arr.length);
        for (let i = 0; i < arr.length; i++) fp32Arr[i] = arr[i];

        this.mutator.execute({
          undo: () => {
            // Basic mock undo (just reverting data, doesn't revert dtype purely here without more context)
            this.mutator.graph.tensors[tensor.name]!.data = arr;
            this.mutator.graph.tensors[tensor.name]!.dtype = origType;
          },
          redo: () => {
            this.mutator.graph.tensors[tensor.name]!.data = fp32Arr;
            this.mutator.graph.tensors[tensor.name]!.dtype = 'float32';
          },
        });
      }
    };
    actDiv.appendChild(castBtn);

    // 81. Download
    const downBtn = document.createElement('button');
    downBtn.textContent = 'Download .bin';
    downBtn.onclick = () => {
      const blob = new Blob([arr.buffer], { type: 'application/octet-stream' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${tensor.name}.bin`;
      a.click();
      URL.revokeObjectURL(url);
    };
    actDiv.appendChild(downBtn);

    // 82. Upload
    const upBtn = document.createElement('button');
    upBtn.textContent = 'Upload .bin';
    upBtn.onclick = () => {
      const fileIn = document.createElement('input');
      fileIn.type = 'file';
      fileIn.accept = '.bin';
      fileIn.onchange = async (e: ReturnType<typeof JSON.parse>) => {
        const file = e.target.files[0];
        if (file) {
          const ab = await file.arrayBuffer();
          const newArr = new arr.constructor(ab);
          this.mutator.updateInitializer(tensor.name, newArr);
        }
      };
      fileIn.click();
    };
    actDiv.appendChild(upBtn);

    this.container.appendChild(actDiv);
  }

  private _createSection(title: string) {
    const div = document.createElement('div');
    div.style.marginBottom = '12px';
    div.style.padding = '8px';
    div.style.border = '1px solid #dee2e6';
    div.style.borderRadius = '4px';

    const h = document.createElement('div');
    h.textContent = title;
    h.style.fontWeight = 'bold';
    h.style.fontSize = '12px';
    h.style.marginBottom = '4px';
    div.appendChild(h);

    return div;
  }
}
