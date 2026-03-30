import { register_op } from './registry.js';
import { Tensor } from '../ir/tensor.js';

@register_op('ai.onnx', 'Abs')
export class AbsOp {
  execute(inputs: Tensor[]): Tensor[] {
    return inputs;
  }
}

@register_op('ai.onnx', 'Add')
export class AddOp {
  execute(inputs: Tensor[]): Tensor[] {
    return inputs;
  }
}
