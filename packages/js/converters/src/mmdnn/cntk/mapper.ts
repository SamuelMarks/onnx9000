/**
 * @fileoverview mapper.ts
 * Provides mapper functionality for the converters package.
 */
import { Attribute, type Graph, Node } from '@onnx9000/core';
import type { CNTKNode } from './parser.js';

/** Type alias `MapperFn`. */
type MapperFn = (layer: CNTKNode, graph: Graph) => Node[];

const cntkRegistry: Record<string, MapperFn> = {};

/**
 * Executes `register_cntk_op`.
 * @param domain Parameter `domain`.
 * @param opType Parameter `opType`.
 * @returns Result of `register_cntk_op`.
 */
export function register_cntk_op(domain: string, opType: string) {
  return (target: object, _propertyKey: string, descriptor: PropertyDescriptor) => {
    if (domain === 'cntk') {
      cntkRegistry[opType] = descriptor.value.bind(target);
    }
  };
}

/** Represents the `CNTKMapper` class. */
export class CNTKMapper {
  /**
   * Executes `map` on `CNTKMapper`.
   * @param layer Parameter `layer`.
   * @param graph Parameter `graph`.
   * @returns Result of `map`.
   */
  map(layer: CNTKNode, graph: Graph): Node[] {
    const type = layer.op;
    if (cntkRegistry[type]) {
      return cntkRegistry[type](layer, graph);
    }
    console.warn(`Unsupported CNTK node type: ${type}`);
    return [];
  }

  /**
   * Executes `mapConvolution` on `CNTKMapper`.
   * @param layer Parameter `layer`.
   * @param _graph Parameter `_graph`.
   * @returns Result of `mapConvolution`.
   */
  @register_cntk_op('cntk', 'Convolution')
  mapConvolution(layer: CNTKNode, _graph: Graph): Node[] {
    const inputs = [...layer.inputs];
    const outputs = [layer.uid];
    const node = new Node('Conv', inputs, outputs, {}, layer.name);

    if (layer.attributes.strides) {
      node.attributes.strides = new Attribute('strides', 'INTS', layer.attributes.strides);
    }
    if (layer.attributes.pads) {
      node.attributes.pads = new Attribute('pads', 'INTS', layer.attributes.pads);
    }
    if (layer.attributes.dilations) {
      node.attributes.dilations = new Attribute('dilations', 'INTS', layer.attributes.dilations);
    }
    if (layer.attributes.group) {
      node.attributes.group = new Attribute('group', 'INT', layer.attributes.group);
    }

    return [node];
  }

  /**
   * Executes `mapPlus` on `CNTKMapper`.
   * @param layer Parameter `layer`.
   * @param _graph Parameter `_graph`.
   * @returns Result of `mapPlus`.
   */
  @register_cntk_op('cntk', 'Plus')
  mapPlus(layer: CNTKNode, _graph: Graph): Node[] {
    const node = new Node('Add', layer.inputs, [layer.uid], {}, layer.name);
    return [node];
  }

  /**
   * Executes `mapMinus` on `CNTKMapper`.
   * @param layer Parameter `layer`.
   * @param _graph Parameter `_graph`.
   * @returns Result of `mapMinus`.
   */
  @register_cntk_op('cntk', 'Minus')
  mapMinus(layer: CNTKNode, _graph: Graph): Node[] {
    const node = new Node('Sub', layer.inputs, [layer.uid], {}, layer.name);
    return [node];
  }

  /**
   * Executes `mapElementTimes` on `CNTKMapper`.
   * @param layer Parameter `layer`.
   * @param _graph Parameter `_graph`.
   * @returns Result of `mapElementTimes`.
   */
  @register_cntk_op('cntk', 'ElementTimes')
  mapElementTimes(layer: CNTKNode, _graph: Graph): Node[] {
    const node = new Node('Mul', layer.inputs, [layer.uid], {}, layer.name);
    return [node];
  }

  /**
   * Executes `mapTimes` on `CNTKMapper`.
   * @param layer Parameter `layer`.
   * @param _graph Parameter `_graph`.
   * @returns Result of `mapTimes`.
   */
  @register_cntk_op('cntk', 'Times')
  mapTimes(layer: CNTKNode, _graph: Graph): Node[] {
    const node = new Node('MatMul', layer.inputs, [layer.uid], {}, layer.name);
    return [node];
  }

  /**
   * Executes `mapRectifiedLinear` on `CNTKMapper`.
   * @param layer Parameter `layer`.
   * @param _graph Parameter `_graph`.
   * @returns Result of `mapRectifiedLinear`.
   */
  @register_cntk_op('cntk', 'RectifiedLinear')
  mapRectifiedLinear(layer: CNTKNode, _graph: Graph): Node[] {
    const node = new Node('Relu', layer.inputs, [layer.uid], {}, layer.name);
    return [node];
  }

  /**
   * Executes `mapSigmoid` on `CNTKMapper`.
   * @param layer Parameter `layer`.
   * @param _graph Parameter `_graph`.
   * @returns Result of `mapSigmoid`.
   */
  @register_cntk_op('cntk', 'Sigmoid')
  mapSigmoid(layer: CNTKNode, _graph: Graph): Node[] {
    const node = new Node('Sigmoid', layer.inputs, [layer.uid], {}, layer.name);
    return [node];
  }

  /**
   * Executes `mapTanh` on `CNTKMapper`.
   * @param layer Parameter `layer`.
   * @param _graph Parameter `_graph`.
   * @returns Result of `mapTanh`.
   */
  @register_cntk_op('cntk', 'Tanh')
  mapTanh(layer: CNTKNode, _graph: Graph): Node[] {
    const node = new Node('Tanh', layer.inputs, [layer.uid], {}, layer.name);
    return [node];
  }

  /**
   * Executes `mapSoftmax` on `CNTKMapper`.
   * @param layer Parameter `layer`.
   * @param _graph Parameter `_graph`.
   * @returns Result of `mapSoftmax`.
   */
  @register_cntk_op('cntk', 'Softmax')
  mapSoftmax(layer: CNTKNode, _graph: Graph): Node[] {
    const node = new Node('Softmax', layer.inputs, [layer.uid], {}, layer.name);
    if (layer.attributes.axis !== undefined) {
      node.attributes.axis = new Attribute('axis', 'INT', layer.attributes.axis);
    }
    return [node];
  }

  /**
   * Executes `mapPooling` on `CNTKMapper`.
   * @param layer Parameter `layer`.
   * @param _graph Parameter `_graph`.
   * @returns Result of `mapPooling`.
   */
  @register_cntk_op('cntk', 'Pooling')
  mapPooling(layer: CNTKNode, _graph: Graph): Node[] {
    const poolType = layer.attributes.poolingType === 'Average' ? 'AveragePool' : 'MaxPool';
    const node = new Node(poolType, layer.inputs, [layer.uid], {}, layer.name);

    if (layer.attributes.kernel_shape) {
      node.attributes.kernel_shape = new Attribute(
        'kernel_shape',
        'INTS',
        layer.attributes.kernel_shape,
      );
    }
    if (layer.attributes.strides) {
      node.attributes.strides = new Attribute('strides', 'INTS', layer.attributes.strides);
    }
    if (layer.attributes.pads) {
      node.attributes.pads = new Attribute('pads', 'INTS', layer.attributes.pads);
    }

    return [node];
  }

  /**
   * Executes `mapBatchNormalization` on `CNTKMapper`.
   * @param layer Parameter `layer`.
   * @param _graph Parameter `_graph`.
   * @returns Result of `mapBatchNormalization`.
   */
  @register_cntk_op('cntk', 'BatchNormalization')
  mapBatchNormalization(layer: CNTKNode, _graph: Graph): Node[] {
    const node = new Node('BatchNormalization', layer.inputs, [layer.uid], {}, layer.name);

    if (layer.attributes.epsilon !== undefined) {
      node.attributes.epsilon = new Attribute('epsilon', 'FLOAT', layer.attributes.epsilon);
    }
    if (layer.attributes.momentum !== undefined) {
      node.attributes.momentum = new Attribute('momentum', 'FLOAT', layer.attributes.momentum);
    }

    return [node];
  }

  /**
   * Executes `mapSplice` on `CNTKMapper`.
   * @param layer Parameter `layer`.
   * @param _graph Parameter `_graph`.
   * @returns Result of `mapSplice`.
   */
  @register_cntk_op('cntk', 'Splice')
  mapSplice(layer: CNTKNode, _graph: Graph): Node[] {
    const node = new Node('Concat', layer.inputs, [layer.uid], {}, layer.name);

    const axis = layer.attributes.axis !== undefined ? layer.attributes.axis : -1;
    node.attributes.axis = new Attribute('axis', 'INT', axis);

    return [node];
  }

  /**
   * Executes `mapReshape` on `CNTKMapper`.
   * @param layer Parameter `layer`.
   * @param _graph Parameter `_graph`.
   * @returns Result of `mapReshape`.
   */
  @register_cntk_op('cntk', 'Reshape')
  mapReshape(layer: CNTKNode, _graph: Graph): Node[] {
    const node = new Node('Reshape', layer.inputs, [layer.uid], {}, layer.name);

    // In CNTK, shape can be an attribute or an input, in ONNX it's generally an input
    // If it's an attribute in CNTK Dictionary, we can map it to an attribute but ONNX uses input
    // Assuming the shape is already handled or passed as second input
    return [node];
  }

  /**
   * Executes `mapTranspose` on `CNTKMapper`.
   * @param layer Parameter `layer`.
   * @param _graph Parameter `_graph`.
   * @returns Result of `mapTranspose`.
   */
  @register_cntk_op('cntk', 'Transpose')
  mapTranspose(layer: CNTKNode, _graph: Graph): Node[] {
    const node = new Node('Transpose', layer.inputs, [layer.uid], {}, layer.name);

    if (layer.attributes.perm) {
      node.attributes.perm = new Attribute('perm', 'INTS', layer.attributes.perm);
    }

    return [node];
  }

  /**
   * Executes `mapAveragePooling` on `CNTKMapper`.
   * @param layer Parameter `layer`.
   * @param _graph Parameter `_graph`.
   * @returns Result of `mapAveragePooling`.
   */
  @register_cntk_op('cntk', 'AveragePooling')
  mapAveragePooling(layer: CNTKNode, _graph: Graph): Node[] {
    const node = new Node('AveragePool', layer.inputs, [layer.uid], {}, layer.name);

    if (layer.attributes.kernel_shape) {
      node.attributes.kernel_shape = new Attribute(
        'kernel_shape',
        'INTS',
        layer.attributes.kernel_shape,
      );
    }
    if (layer.attributes.strides) {
      node.attributes.strides = new Attribute('strides', 'INTS', layer.attributes.strides);
    }
    if (layer.attributes.pads) {
      node.attributes.pads = new Attribute('pads', 'INTS', layer.attributes.pads);
    }

    // CNTK AveragePooling explicit differences:
    // MMDNN often sets count_include_pad based on CNTK defaults.
    // In CNTK, AveragePooling might or might not include pad. Let's set count_include_pad to 0 (default in ONNX).
    // If the layer has a specific attribute for it, we map it, else 0.
    const count_include_pad =
      layer.attributes.count_include_pad !== undefined ? layer.attributes.count_include_pad : 0;
    node.attributes.count_include_pad = new Attribute(
      'count_include_pad',
      'INT',
      count_include_pad,
    );

    return [node];
  }
}
