import { OnnxNodeBuilder } from './emitters.js';

export function mapTfjsOpToOnnx(
  opType: string,
  inputs: string[],
  outputName: string,
  name: string,
): OnnxNodeBuilder[] {
  const nodes: OnnxNodeBuilder[] = [];

  let onnxOp = '';
  const attributes: {
    name: string;
    type?: string;
    f?: number;
    i?: number;
    ints?: number[];
    floats?: number[];
    s?: string;
  }[] = [];

  switch (opType) {
    case 'Add':
    case 'AddV2':
      onnxOp = 'Add';
      break;
    case 'Sub':
      onnxOp = 'Sub';
      break;
    case 'Mul':
      onnxOp = 'Mul';
      break;
    case 'RealDiv':
    case 'Div':
      onnxOp = 'Div';
      break;
    case 'MatMul':
      onnxOp = 'MatMul';
      break;
    case 'Square':
      onnxOp = 'Pow';
      /* needs exponent 2 input in ONNX but simplified here */ break;
    case 'Sqrt':
      onnxOp = 'Sqrt';
      break;
    case 'Exp':
      onnxOp = 'Exp';
      break;
    case 'Log':
      onnxOp = 'Log';
      break;
    case 'Maximum':
      onnxOp = 'Max';
      break;
    case 'Minimum':
      onnxOp = 'Min';
      break;
    case 'Sum':
      onnxOp = 'ReduceSum';
      break;
    case 'Mean':
      onnxOp = 'ReduceMean';
      break;
    case 'Max':
      onnxOp = 'ReduceMax';
      break;
    case 'Min':
      onnxOp = 'ReduceMin';
      break;
    case 'ArgMax':
      onnxOp = 'ArgMax';
      break;
    case 'ArgMin':
      onnxOp = 'ArgMin';
      break;
    case 'Split':
    case 'SplitV':
      onnxOp = 'Split';
      break;
    case 'Concat':
    case 'ConcatV2':
      onnxOp = 'Concat';
      break;
    case 'Slice':
      onnxOp = 'Slice';
      break;
    case 'StridedSlice':
      onnxOp = 'Slice';
      break; // Requires complex attribute mapping
    case 'Gather':
    case 'GatherV2':
      onnxOp = 'Gather';
      break;
    case 'GatherNd':
      onnxOp = 'GatherND';
      break;
    case 'Where':
      onnxOp = 'Where';
      break;
    case 'TensorScatterUpdate':
      onnxOp = 'ScatterND';
      break;
    case 'ResizeBilinear':
      onnxOp = 'Resize';
      attributes.push({ name: 'mode', s: 'linear', type: 'STRING' });
      break;
    case 'ResizeNearestNeighbor':
      onnxOp = 'Resize';
      attributes.push({ name: 'mode', s: 'nearest', type: 'STRING' });
      break;
    default:
      throw new Error(`Unsupported TF.js Op: ${opType}`);
  }

  nodes.push({
    opType: onnxOp,
    inputs,
    outputs: [outputName],
    name,
    attributes,
  });

  return nodes;
}
