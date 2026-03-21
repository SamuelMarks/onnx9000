import { Graph, Node, Attribute } from '@onnx9000/core';
import { CNTKNode } from './parser.js';

type MapperFn = (layer: CNTKNode, graph: Graph) => Node[];

const cntkRegistry: Record<string, MapperFn> = {};

export function register_cntk_op(domain: string, opType: string) {
  return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    if (domain === 'cntk') {
      cntkRegistry[opType] = descriptor.value.bind(target);
    }
  };
}

export class CNTKMapper {
  map(layer: CNTKNode, graph: Graph): Node[] {
    const type = layer.op;
    if (cntkRegistry[type]) {
      return cntkRegistry[type](layer, graph);
    }
    console.warn(`Unsupported CNTK node type: ${type}`);
    return [];
  }

  @register_cntk_op('cntk', 'Convolution')
  mapConvolution(layer: CNTKNode, graph: Graph): Node[] {
    const inputs = [...layer.inputs];
    const outputs = [layer.uid];
    const node = new Node('Conv', inputs, outputs, {}, layer.name);

    if (layer.attributes.strides) {
      node.attributes['strides'] = new Attribute('strides', 'INTS', layer.attributes.strides);
    }
    if (layer.attributes.pads) {
      node.attributes['pads'] = new Attribute('pads', 'INTS', layer.attributes.pads);
    }
    if (layer.attributes.dilations) {
      node.attributes['dilations'] = new Attribute('dilations', 'INTS', layer.attributes.dilations);
    }
    if (layer.attributes.group) {
      node.attributes['group'] = new Attribute('group', 'INT', layer.attributes.group);
    }

    return [node];
  }

  @register_cntk_op('cntk', 'Plus')
  mapPlus(layer: CNTKNode, graph: Graph): Node[] {
    const node = new Node('Add', layer.inputs, [layer.uid], {}, layer.name);
    return [node];
  }

  @register_cntk_op('cntk', 'Minus')
  mapMinus(layer: CNTKNode, graph: Graph): Node[] {
    const node = new Node('Sub', layer.inputs, [layer.uid], {}, layer.name);
    return [node];
  }

  @register_cntk_op('cntk', 'ElementTimes')
  mapElementTimes(layer: CNTKNode, graph: Graph): Node[] {
    const node = new Node('Mul', layer.inputs, [layer.uid], {}, layer.name);
    return [node];
  }

  @register_cntk_op('cntk', 'Times')
  mapTimes(layer: CNTKNode, graph: Graph): Node[] {
    const node = new Node('MatMul', layer.inputs, [layer.uid], {}, layer.name);
    return [node];
  }

  @register_cntk_op('cntk', 'RectifiedLinear')
  mapRectifiedLinear(layer: CNTKNode, graph: Graph): Node[] {
    const node = new Node('Relu', layer.inputs, [layer.uid], {}, layer.name);
    return [node];
  }

  @register_cntk_op('cntk', 'Sigmoid')
  mapSigmoid(layer: CNTKNode, graph: Graph): Node[] {
    const node = new Node('Sigmoid', layer.inputs, [layer.uid], {}, layer.name);
    return [node];
  }

  @register_cntk_op('cntk', 'Tanh')
  mapTanh(layer: CNTKNode, graph: Graph): Node[] {
    const node = new Node('Tanh', layer.inputs, [layer.uid], {}, layer.name);
    return [node];
  }

  @register_cntk_op('cntk', 'Softmax')
  mapSoftmax(layer: CNTKNode, graph: Graph): Node[] {
    const node = new Node('Softmax', layer.inputs, [layer.uid], {}, layer.name);
    if (layer.attributes.axis !== undefined) {
      node.attributes['axis'] = new Attribute('axis', 'INT', layer.attributes.axis);
    }
    return [node];
  }

  @register_cntk_op('cntk', 'Pooling')
  mapPooling(layer: CNTKNode, graph: Graph): Node[] {
    const poolType = layer.attributes.poolingType === 'Average' ? 'AveragePool' : 'MaxPool';
    const node = new Node(poolType, layer.inputs, [layer.uid], {}, layer.name);

    if (layer.attributes.kernel_shape) {
      node.attributes['kernel_shape'] = new Attribute(
        'kernel_shape',
        'INTS',
        layer.attributes.kernel_shape,
      );
    }
    if (layer.attributes.strides) {
      node.attributes['strides'] = new Attribute('strides', 'INTS', layer.attributes.strides);
    }
    if (layer.attributes.pads) {
      node.attributes['pads'] = new Attribute('pads', 'INTS', layer.attributes.pads);
    }

    return [node];
  }

  @register_cntk_op('cntk', 'BatchNormalization')
  mapBatchNormalization(layer: CNTKNode, graph: Graph): Node[] {
    const node = new Node('BatchNormalization', layer.inputs, [layer.uid], {}, layer.name);

    if (layer.attributes.epsilon !== undefined) {
      node.attributes['epsilon'] = new Attribute('epsilon', 'FLOAT', layer.attributes.epsilon);
    }
    if (layer.attributes.momentum !== undefined) {
      node.attributes['momentum'] = new Attribute('momentum', 'FLOAT', layer.attributes.momentum);
    }

    return [node];
  }

  @register_cntk_op('cntk', 'Splice')
  mapSplice(layer: CNTKNode, graph: Graph): Node[] {
    const node = new Node('Concat', layer.inputs, [layer.uid], {}, layer.name);

    const axis = layer.attributes.axis !== undefined ? layer.attributes.axis : -1;
    node.attributes['axis'] = new Attribute('axis', 'INT', axis);

    return [node];
  }

  @register_cntk_op('cntk', 'Reshape')
  mapReshape(layer: CNTKNode, graph: Graph): Node[] {
    const node = new Node('Reshape', layer.inputs, [layer.uid], {}, layer.name);

    // In CNTK, shape can be an attribute or an input, in ONNX it's generally an input
    // If it's an attribute in CNTK Dictionary, we can map it to an attribute but ONNX uses input
    // Assuming the shape is already handled or passed as second input
    return [node];
  }

  @register_cntk_op('cntk', 'Transpose')
  mapTranspose(layer: CNTKNode, graph: Graph): Node[] {
    const node = new Node('Transpose', layer.inputs, [layer.uid], {}, layer.name);

    if (layer.attributes.perm) {
      node.attributes['perm'] = new Attribute('perm', 'INTS', layer.attributes.perm);
    }

    return [node];
  }

  @register_cntk_op('cntk', 'AveragePooling')
  mapAveragePooling(layer: CNTKNode, graph: Graph): Node[] {
    const node = new Node('AveragePool', layer.inputs, [layer.uid], {}, layer.name);

    if (layer.attributes.kernel_shape) {
      node.attributes['kernel_shape'] = new Attribute(
        'kernel_shape',
        'INTS',
        layer.attributes.kernel_shape,
      );
    }
    if (layer.attributes.strides) {
      node.attributes['strides'] = new Attribute('strides', 'INTS', layer.attributes.strides);
    }
    if (layer.attributes.pads) {
      node.attributes['pads'] = new Attribute('pads', 'INTS', layer.attributes.pads);
    }

    // CNTK AveragePooling explicit differences:
    // MMDNN often sets count_include_pad based on CNTK defaults.
    // In CNTK, AveragePooling might or might not include pad. Let's set count_include_pad to 0 (default in ONNX).
    // If the layer has a specific attribute for it, we map it, else 0.
    const count_include_pad =
      layer.attributes.count_include_pad !== undefined ? layer.attributes.count_include_pad : 0;
    node.attributes['count_include_pad'] = new Attribute(
      'count_include_pad',
      'INT',
      count_include_pad,
    );

    return [node];
  }
}
