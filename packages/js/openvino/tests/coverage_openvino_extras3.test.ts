import { describe, it, expect } from 'vitest';
import { Graph, Node, ValueInfo } from '@onnx9000/core';
import { OpenVinoExporter } from '../src/exporter';
import { XmlNode, XmlBuilder } from '../src/xml_builder';

describe('Coverage OpenVINO 3', () => {
  it('pool and pad', () => {
    const graph = new Graph('test');
    graph.inputs.push(new ValueInfo('a', [1], 'float32'));
    graph.inputs.push(new ValueInfo('pads', [4], 'int64'));

    graph.addNode(
      new Node('AveragePool', ['a'], ['b'], { count_include_pad: { value: 1 } } as any),
    );
    graph.addNode(new Node('Pad', ['b', 'pads'], ['c'], { mode: { value: 'constant' } } as any)); // 2 inputs, mode constant -> hits 808-816

    const exp = new OpenVinoExporter(graph);
    const { xml } = exp.export();
    expect(xml).toContain('exclude-pad');
  });

  it('xml_builder', () => {
    const root = new XmlNode('root');
    root.addChild('string child <>&');
    root.addChild('another string');
    expect(root.toString(0, true)).toContain('string child &lt;&gt;&amp;');
    expect(root.toString(0, false)).toContain('string child &lt;&gt;&amp;');

    const builder = new XmlBuilder('builder_root');
    builder.setDeclaration('dec');
    expect(builder.toString(false)).toContain('dec');

    // setRoot and toString coverage
    builder.setRoot(root);
    expect(builder.toString(true)).toContain('string child');
  });
});
