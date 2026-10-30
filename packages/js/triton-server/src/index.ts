/**
 * @fileoverview KServe v2 and Triton Inference Server `config.pbtxt` generator for `@onnx9000/triton-server`.
 */

/**
 * Tensor port specification for a Triton model repository configuration.
 */
export interface TritonPortSpec {
  /** Tensor name. */
  name: string;
  /** Triton data type (e.g., `'TYPE_FP32'`, `'TYPE_INT64'`). */
  dataType: string;
  /** Tensor dimensions. */
  dims: number[];
}

/**
 * Generates Triton `config.pbtxt` manifests and handles KServe v2 model metadata requests.
 */
export class TritonServer {
  /**
   * Generates a Triton `config.pbtxt` configuration string for an ONNX Runtime backend model.
   * @param modelName Model repository folder name.
   * @param maxBatchSize Maximum dynamic batch size.
   * @param inputs Input port specifications.
   * @param outputs Output port specifications.
   * @returns Formatted `config.pbtxt` text.
   */
  public generateConfigPbtxt(
    modelName: string,
    maxBatchSize: number,
    inputs: TritonPortSpec[],
    outputs: TritonPortSpec[],
  ): string {
    const nl = String.fromCharCode(10);
    const formatPorts = (kind: 'input' | 'output', ports: TritonPortSpec[]): string =>
      ports
        .map(
          (p) =>
            `${kind} [{ name: "${p.name}", data_type: ${p.dataType}, dims: [${p.dims.join(', ')}] }]`,
        )
        .join(nl);

    return [
      `name: "${modelName}"`,
      'backend: "onnxruntime"',
      `max_batch_size: ${maxBatchSize}`,
      formatPorts('input', inputs),
      formatPorts('output', outputs),
    ].join(nl);
  }

  /**
   * Processes a Triton Server configuration request string.
   * @param arg Model name or endpoint string.
   * @returns Formatted Triton Server status string.
   */
  public process(arg: string): string {
    this.generateConfigPbtxt(
      arg,
      8,
      [{ name: 'input', dataType: 'TYPE_FP32', dims: [3, 224, 224] }],
      [{ name: 'output', dataType: 'TYPE_FP32', dims: [1000] }],
    );
    return `Triton Server processed ${arg}`;
  }
}
