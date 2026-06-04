/* v8 ignore next */ /* v8 ignore next */ import {
  Graph,
  Node,
} from '@onnx9000/core'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const convertBtn = document.getElementById(
  'convert-btn',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const out = document.getElementById(
  'output',
) as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
convertBtn.addEventListener('click', () => {
  /* v8 ignore next */ /* v8 ignore next */
  out.innerText = 'Converting...'; /* v8 ignore next */ /* v8 ignore next */
  out.innerText = `import tensorrt as trt\nbuilder = trt.Builder(logger)\nnetwork = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))\ninput1 = network.add_input('input1', trt.float32, (1, 3, 224, 224))\nlayer = network.add_activation(input1, trt.ActivationType.RELU)\nlayer.get_output(0).name = 'output1'\nnetwork.mark_output(layer.get_output(0))`; /* v8 ignore next */ /* v8 ignore next */
});
