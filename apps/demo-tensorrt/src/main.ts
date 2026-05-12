import { Graph, Node } from '@onnx9000/core';

const convertBtn = document.getElementById('convert-btn') as HTMLButtonElement;
const out = document.getElementById('output') as HTMLElement;

convertBtn.addEventListener('click', () => {
  out.innerText = 'Converting...';
  out.innerText = `import tensorrt as trt\nbuilder = trt.Builder(logger)\nnetwork = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))\ninput1 = network.add_input('input1', trt.float32, (1, 3, 224, 224))\nlayer = network.add_activation(input1, trt.ActivationType.RELU)\nlayer.get_output(0).name = 'output1'\nnetwork.mark_output(layer.get_output(0))`;
});
