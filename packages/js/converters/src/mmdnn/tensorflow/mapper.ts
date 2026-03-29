/* eslint-disable */
// @ts-nocheck
import { Graph, Node, Attribute, Tensor } from '@onnx9000/core';
import { TFNodeDef } from './parser.js';

/**
 * Maps TensorFlow NodeDefs to ONNX Nodes.
 */
export class TFMapper {
  /**
   * Translates a parsed TensorFlow node into an equivalent array of ONNX nodes.
   *
   * @param node The TensorFlow node definition parsed from a .pbtxt file.
   * @param graph The ONNX graph to append initializers and tensors to.
   * @returns An array of translated ONNX nodes.
   */
  map(node: TFNodeDef, graph: Graph): Node[] {
    const attrs: Record<string, Attribute> = {};
    const outputs = [node.name];
    let opType = node.op;

    // Convert generic attributes
    for (const [key, attr] of Object.entries(node.attr)) {
      if (attr.list && attr.list.i) {
        attrs[key] = new Attribute(key, 'INTS', attr.list.i);
      } else if (attr.i !== undefined) {
        attrs[key] = new Attribute(key, 'INT', attr.i);
      } else if (attr.f !== undefined) {
        attrs[key] = new Attribute(key, 'FLOAT', attr.f);
      } else if (attr.s !== undefined) {
        attrs[key] = new Attribute(key, 'STRING', attr.s);
      } else if (attr.shape) {
        attrs[key] = new Attribute(key, 'INTS', attr.shape);
      }
    }

    // Special op mappings
    if (opType === 'Placeholder') {
      opType = 'Identity';
      const outNode = new Node(
        opType,
        node.input.length ? node.input : [node.name + '_input_dummy'],
        outputs,
        attrs,
        node.name,
      );
      return [outNode];
    } else if (opType === 'Const') {
      // In ONNX, Constants are usually initializers
      opType = 'Constant';
      graph.initializers.push(node.name);

      const tfAttr = node.attr['value'];
      let shape = [1];
      if (tfAttr && tfAttr.tensor && tfAttr.tensor.shape && tfAttr.tensor.shape.length > 0) {
        shape = tfAttr.tensor.shape;
      }

      const tensor = new Tensor(node.name, shape, 'float32');
      // Create empty buffer
      const size = shape.reduce((a, b) => a * Math.abs(b), 1) || 1;
      tensor.data = new Uint8Array(size * 4);
      graph.tensors[node.name] = tensor;

      return []; // Don't add a node, it's an initializer
    } else if (opType === 'Relu6') {
      opType = 'Relu'; // Simplified fallback for Relu6
    } else if (opType === 'Conv2D' || opType === 'DepthwiseConv2dNative') {
      opType = 'Conv';

      // TensorFlow uses [1, stride_h, stride_w, 1], ONNX expects [stride_h, stride_w]
      if (
        attrs['strides'] &&
        attrs['strides'].type === 'INTS' &&
        Array.isArray(attrs['strides'].value) &&
        attrs['strides'].value.length === 4
      ) {
        const tfStrides = attrs['strides'].value as number[];
        attrs['strides'] = new Attribute('strides', 'INTS', [tfStrides[1], tfStrides[2]]);
      }

      if (attrs['padding'] && attrs['padding'].type === 'STRING') {
        const padStr = attrs['padding'].value;
        if (padStr === 'SAME') {
          attrs['auto_pad'] = new Attribute('auto_pad', 'STRING', 'SAME_UPPER');
        } else if (padStr === 'VALID') {
          attrs['auto_pad'] = new Attribute('auto_pad', 'STRING', 'VALID');
        }
        delete attrs['padding'];
      }
    } else if (opType === 'MaxPool' || opType === 'AvgPool') {
      if (
        attrs['ksize'] &&
        attrs['ksize'].type === 'INTS' &&
        Array.isArray(attrs['ksize'].value) &&
        attrs['ksize'].value.length === 4
      ) {
        const ksize = attrs['ksize'].value as number[];
        attrs['kernel_shape'] = new Attribute('kernel_shape', 'INTS', [ksize[1], ksize[2]]);
        delete attrs['ksize'];
      }
      if (
        attrs['strides'] &&
        attrs['strides'].type === 'INTS' &&
        Array.isArray(attrs['strides'].value) &&
        attrs['strides'].value.length === 4
      ) {
        const tfStrides = attrs['strides'].value as number[];
        attrs['strides'] = new Attribute('strides', 'INTS', [tfStrides[1], tfStrides[2]]);
      }
      if (attrs['padding'] && attrs['padding'].type === 'STRING') {
        const padStr = attrs['padding'].value;
        if (padStr === 'SAME') {
          attrs['auto_pad'] = new Attribute('auto_pad', 'STRING', 'SAME_UPPER');
        } else if (padStr === 'VALID') {
          attrs['auto_pad'] = new Attribute('auto_pad', 'STRING', 'VALID');
        }
        delete attrs['padding'];
      }
    }

    const outNode = new Node(opType, node.input, outputs, attrs, node.name);
    return [outNode];
  }
}
