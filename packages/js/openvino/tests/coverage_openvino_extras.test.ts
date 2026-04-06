import { describe, it, expect, vi } from 'vitest';
import { Graph, Node, Tensor, ValueInfo } from '@onnx9000/core';
import { OpenVinoExporter } from '../src/exporter';
import { main as runCli } from '../bin/cli.js'; // to test cli

describe('Coverage OpenVINO', () => {
  it('exhausts all operator handlers', () => {
    const graph = new Graph('test');

    graph.inputs.push(new ValueInfo('a', [1], 'float32'));
    graph.inputs.push(new ValueInfo('b', [1], 'float32'));
    graph.inputs.push(new ValueInfo('pad_val', [1], 'float32'));
    graph.inputs.push(new ValueInfo('idx', [1], 'int64'));
    graph.inputs.push(new ValueInfo('idx2', [1], 'int64'));

    graph.addNode(
      new Node('MatMul', ['a', 'b'], ['c'], { transA: { value: 1 }, transB: { value: 1 } } as any),
    );
    graph.addNode(
      new Node('Conv', ['c'], ['d'], {
        strides: { value: [1, 1] },
        pads: { value: [1, 1, 1, 1] },
        auto_pad: { value: 'SAME_UPPER' },
      } as any),
    );
    graph.addNode(
      new Node('MaxPool', ['d'], ['e'], {
        strides: { value: [1, 1] },
        kernel_shape: { value: [2, 2] },
        pads: { value: [0, 0, 0, 0] },
      } as any),
    );
    graph.addNode(new Node('Gelu', ['e'], ['f'], { approximate: { value: 'tanh' } } as any));
    graph.addNode(new Node('Softmax', ['f'], ['g'], { axis: { value: 1 } } as any));
    graph.addNode(
      new Node('Pad', ['g', 'pad_val'], ['h'], {
        mode: { value: 'reflect' },
        pads: { value: [1, 1, 1, 1] },
      } as any),
    );
    graph.addNode(new Node('Gather', ['h', 'idx'], ['i'], { axis: { value: 1 } } as any));
    graph.addNode(new Node('Slice', ['i'], ['j'], {} as any));
    graph.addNode(
      new Node('ReduceMean', ['j'], ['k'], { keepdims: { value: 1 }, axes: { value: [0] } } as any),
    );
    graph.addNode(
      new Node('ArgMax', ['k'], ['l'], { axis: { value: 1 }, keepdims: { value: 1 } } as any),
    );
    graph.addNode(
      new Node('Resize', ['l'], ['m'], {
        mode: { value: 'linear' },
        coordinate_transformation_mode: { value: 'pytorch_half_pixel' },
      } as any),
    );
    graph.addNode(new Node('SpaceToDepth', ['m'], ['n'], { blocksize: { value: 2 } } as any));
    graph.addNode(
      new Node('NonMaxSuppression', ['n'], ['o'], { center_point_box: { value: 1 } } as any),
    );
    graph.addNode(new Node('RoiAlign', ['o'], ['p'], { mode: { value: 'avg' } } as any));
    graph.addNode(new Node('QuantizeLinear', ['p'], ['q'], {} as any));
    graph.addNode(new Node('Einsum', ['q'], ['r'], { equation: { value: 'a,b->c' } } as any));
    graph.addNode(
      new Node('LayerNormalization', ['r'], ['s'], { epsilon: { value: 1e-5 } } as any),
    );
    graph.addNode(
      new Node('BatchNormalization', ['s'], ['t'], { epsilon: { value: 1e-5 } } as any),
    );

    // Complex nodes with subgraphs
    const ifNode = new Node('If', ['t'], ['u']);
    ifNode.attributes['then_branch'] = { value: new Graph('then') } as any;
    ifNode.attributes['else_branch'] = { value: new Graph('else') } as any;
    graph.addNode(ifNode);

    const loopNode = new Node('Loop', ['u'], ['v']);
    loopNode.attributes['body'] = { value: new Graph('body') } as any;
    graph.addNode(loopNode);

    graph.addNode(new Node('Cast', ['v'], ['w'], { to: { value: 'float32' } } as any));
    graph.addNode(
      new Node('GridSample', ['w'], ['x'], {
        align_corners: { value: 1 },
        mode: { value: 'bilinear' },
        padding_mode: { value: 'zeros' },
      } as any),
    );
    graph.addNode(new Node('Size', ['x'], ['y'], {} as any));
    graph.addNode(new Node('Flatten', ['y'], ['z'], { axis: { value: 1 } } as any));
    graph.addNode(new Node('Transpose', ['z'], ['aa'], { perm: { value: [1, 0] } } as any));
    graph.addNode(
      new Node('GatherElements', ['aa', 'idx2'], ['bb'], { axis: { value: 1 } } as any),
    );

    const tVal = new Tensor('val', [1], 'float32', false, true, new Float32Array([1]));
    graph.addNode(new Node('ConstantOfShape', ['bb'], ['cc'], { value: { value: tVal } } as any));

    // Edge cases like pad length not 4
    graph.addNode(
      new Node('Conv', ['c'], ['d2'], {
        pads: { value: [1, 1] },
        dilations: { value: [1, 1] },
      } as any),
    );
    graph.addNode(new Node('Pad', ['g'], ['h2'], { mode: { value: 'constant' } } as any));
    graph.addNode(new Node('ReduceMean', ['j'], ['k2'], { keepdims: { value: 0 } } as any));

    const exp = new OpenVinoExporter(graph);
    const { xml, bin } = exp.export();
    expect(xml).toContain('<net');
  });

  it('dynamic constants coverage', () => {
    const graph = new Graph('test');
    const exp = new OpenVinoExporter(graph);
    const floatData = [1.5, 2.5];
    exp.emitDynamicConst('float', floatData, [2], 'float32');
    const int32Data = [1, 2];
    exp.emitDynamicConst('int32', int32Data, [2], 'int32');
    const int64Data = [1n, 2n];
    exp.emitDynamicConst('int64', int64Data, [2], 'int64');
    expect(() => exp.emitDynamicConst('bool', [true], [1], 'bool')).toThrow(
      'Unsupported dtype for dynamic const: bool',
    );

    // hit cache
    exp.emitDynamicConst('float_cache', floatData, [2], 'float32');
  });

  it('cli coverage', async () => {
    // Just mock process.argv and see if it runs
    const origArgv = process.argv;
    process.argv = [
      'node',
      'cli.js',
      'input.onnx',
      '-o',
      'outdir',
      '--fp16',
      '--dynamic-batch',
      '--shape',
      'a:[1,2,3]',
    ];

    // Mock exporter and fs
    const fs = require('fs');
    const origWrite = fs.writeFileSync;
    const origRead = fs.readFileSync;
    const origExit = process.exit;

    fs.writeFileSync = () => {};
    fs.readFileSync = () => Buffer.from([1, 2, 3]);

    let exited = false;
    process.exit = () => {
      exited = true;
      throw new Error('EXIT');
    };

    // Since load is imported in bin/cli.js directly, we can mock @onnx9000/core
    vi.mock('@onnx9000/core', async () => {
      const actual = (await vi.importActual('@onnx9000/core')) as any;
      return {
        ...actual,
        load: () => {
          const g = new Graph('test');
          g.inputs.push(new ValueInfo('a', [1], 'float32'));
          return g;
        },
      };
    });

    try {
      await runCli();
    } catch (e) {}

    // Help mode
    process.argv = ['node', 'cli.js', '--help'];
    try {
      await runCli();
    } catch (e) {}
    expect(exited).toBe(true);

    fs.writeFileSync = origWrite;
    fs.readFileSync = origRead;
    process.argv = origArgv;
    process.exit = origExit;
  });

  it('index and api and xml builder', async () => {
    const idx = await import('../src/index');
    expect(idx).toBeDefined();

    const api = await import('../src/api');
    const g = new Graph('test');
    expect(api.exportModel(g).xml).toBeDefined();

    const b = await import('../src/xml_builder');
    const node = new b.XmlNode('test');
    node.setAttribute('a', '1');
    node.setAttribute('a', '2'); // overwrite
    node.setAttribute('b', '2');
    node.addChild(new b.XmlNode('child'));
    expect(node.children.length).toBe(1);
  });
});
