/**
 * @fileoverview TensorRT Demo UI.
 */

/**
 * Initializes the TensorRT demo UI.
 */
export function initTensorrtDemo(): void {
  const convertBtn = document.getElementById('convert-btn') as HTMLButtonElement | null;
  const out = document.getElementById('output') as HTMLElement | null;
  if (!convertBtn || !out) return;

  convertBtn.addEventListener('click', () => {
    out.innerText = 'Converting...';
    out.innerText = `import tensorrt as trt
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
input1 = network.add_input('input1', trt.float32, (1, 3, 224, 224))
layer = network.add_activation(input1, trt.ActivationType.RELU)
layer.get_output(0).name = 'output1'
network.mark_output(layer.get_output(0))`;
  });
}

initTensorrtDemo();
