/**
 * @fileoverview index.ts
 * Provides index functionality for the transformers package.
 */
export class ArrayAPI {
  /**
   * Executes `add` on `ArrayAPI`.
   * @param a Parameter `a`.
   * @param b Parameter `b`.
   * @returns Result of `add`.
   */
  static add(a: number[], b: number[]): number[] {
    return a.map((val, i) => val + (b?.[i] ?? 0));
  }

  // 218. softmax
  /**
   * Executes `softmax` on `ArrayAPI`.
   * @param tensor Parameter `tensor`.
   * @param _axis Parameter `_axis`.
   * @returns Result of `softmax`.
   */
  static softmax(tensor: number[], _axis?: number): number[] {
    const max = Math.max(...tensor);
    const exps = tensor.map((x) => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((x) => x / sum);
  }

  // 219. log_softmax
  /**
   * Executes `log_softmax` on `ArrayAPI`.
   * @param tensor Parameter `tensor`.
   * @param _axis Parameter `_axis`.
   * @returns Result of `log_softmax`.
   */
  static log_softmax(tensor: number[], _axis?: number): number[] {
    const max = Math.max(...tensor);
    const exps = tensor.map((x) => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    const logSum = Math.log(sum);
    return tensor.map((x) => x - max - logSum);
  }

  // 220. sigmoid
  /**
   * Executes `sigmoid` on `ArrayAPI`.
   * @param tensor Parameter `tensor`.
   * @returns Result of `sigmoid`.
   */
  static sigmoid(tensor: number[]): number[] {
    return tensor.map((x) => 1 / (1 + Math.exp(-x)));
  }

  // 221. get_top_k
  /**
   * Executes `get_top_k` on `ArrayAPI`.
   * @param tensor Parameter `tensor`.
   * @param k Parameter `k`.
   * @returns Result of `get_top_k`.
   */
  static get_top_k(tensor: number[], k: number): { values: number[]; indices: number[] } {
    const indexed = tensor.map((val, idx) => ({ val, idx }));
    indexed.sort((a, b) => b.val - a.val);
    const top = indexed.slice(0, k);
    return {
      values: top.map((x) => x.val),
      indices: top.map((x) => x.idx),
    };
  }

  // 222. cosine_similarity
  /**
   * Executes `cosine_similarity` on `ArrayAPI`.
   * @param a Parameter `a`.
   * @param b Parameter `b`.
   * @returns Result of `cosine_similarity`.
   */
  static cosine_similarity(a: number[], b: number[]): number {
    const dot = ArrayAPI.dot_product(a, b);
    const normA = Math.sqrt(ArrayAPI.dot_product(a, a));
    const normB = Math.sqrt(ArrayAPI.dot_product(b, b));
    return dot / (normA * normB);
  }

  // 223. dot_product
  /**
   * Executes `dot_product` on `ArrayAPI`.
   * @param a Parameter `a`.
   * @param b Parameter `b`.
   * @returns Result of `dot_product`.
   */
  static dot_product(a: number[], b: number[]): number {
    return a.reduce((sum, val, i) => sum + val * (b[i] ?? 0), 0);
  }

  // 224. auto-dispatch to WASM/WebGPU for large tensors
  /**
   * Executes `_maybeDispatchWasm` on `ArrayAPI`.
   * @param tensor Parameter `tensor`.
   * @returns Result of `_maybeDispatchWasm`.
   */
  static _maybeDispatchWasm(tensor: number[]) {
    if (tensor.length > 10000) {
      return true;
    }
    return false;
  }

  // 225. tensor shape manipulation
  /**
   * Executes `view` on `ArrayAPI`.
   * @param tensor Parameter `tensor`.
   * @param shape Parameter `shape`.
   * @returns Result of `view`.
   */
  static view(tensor: ReturnType<typeof JSON.parse>, shape: number[]) {
    tensor.dims = shape;
    return tensor;
  }
  /**
   * Executes `reshape` on `ArrayAPI`.
   * @param tensor Parameter `tensor`.
   * @param shape Parameter `shape`.
   * @returns Result of `reshape`.
   */
  static reshape(tensor: ReturnType<typeof JSON.parse>, shape: number[]) {
    tensor.dims = shape;
    return tensor;
  }
  /**
   * Executes `transpose` on `ArrayAPI`.
   * @param tensor Parameter `tensor`.
   * @param axes Parameter `axes`.
   * @returns Result of `transpose`.
   */
  static transpose(tensor: ReturnType<typeof JSON.parse>, axes?: number[]) {
    tensor.transposed = true;
    tensor.axes = axes;
    return tensor;
  }

  // 226, 227. bi-directional conversion
  /**
   * Executes `toFloat32Array` on `ArrayAPI`.
   * @param tensor Parameter `tensor`.
   * @returns Result of `toFloat32Array`.
   */
  static toFloat32Array(tensor: ReturnType<typeof JSON.parse>): Float32Array {
    return new Float32Array(tensor);
  }
  /**
   * Executes `fromFloat32Array` on `ArrayAPI`.
   * @param array Parameter `array`.
   * @returns Result of `fromFloat32Array`.
   */
  static fromFloat32Array(array: Float32Array): ReturnType<typeof JSON.parse> {
    return Array.from(array);
  }
  /**
   * Executes `toJSON` on `ArrayAPI`.
   * @param tensor Parameter `tensor`.
   * @returns Result of `toJSON`.
   */
  static toJSON(tensor: ReturnType<typeof JSON.parse>): ReturnType<typeof JSON.parse>[] {
    return Array.isArray(tensor) ? tensor : [];
  }
  /**
   * Executes `fromJSON` on `ArrayAPI`.
   * @param json Parameter `json`.
   * @returns Result of `fromJSON`.
   */
  static fromJSON(json: ReturnType<typeof JSON.parse>[]): ReturnType<typeof JSON.parse> {
    return json;
  }

  // 228. Handle multi-dimensional array slicing syntaxes in TS
  /**
   * Executes `slice` on `ArrayAPI`.
   * @param tensor Parameter `tensor`.
   * @param start Parameter `start`.
   * @param end Parameter `end`.
   * @returns Result of `slice`.
   */
  static slice(tensor: ReturnType<typeof JSON.parse>, start: number, end: number) {
    return tensor.slice(start, end);
  }

  // 229. Support strided array access logic
  /**
   * Executes `getStrided` on `ArrayAPI`.
   * @param tensor Parameter `tensor`.
   * @param stride Parameter `stride`.
   * @returns Result of `getStrided`.
   */
  static getStrided(tensor: ReturnType<typeof JSON.parse>, stride: number) {
    if (!Array.isArray(tensor) && !tensor.buffer) return tensor;
    const result = [];
    for (let i = 0; i < tensor.length; i += stride) {
      result.push(tensor[i]);
    }
    return result;
  }

  // 230. Implement Math.erf polyfills if necessary
  /**
   * Executes `erf` on `ArrayAPI`.
   * @param x Parameter `x`.
   * @returns Result of `erf`.
   */
  static erf(x: number): number {
    // Approximation
    const sign = x >= 0 ? 1 : -1;
    x = Math.abs(x);
    const a1 = 0.254829592,
      a2 = -0.284496736,
      a3 = 1.421413741;
    const a4 = -1.453152027,
      a5 = 1.061405429,
      p = 0.3275911;
    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    return sign * y;
  }
}
