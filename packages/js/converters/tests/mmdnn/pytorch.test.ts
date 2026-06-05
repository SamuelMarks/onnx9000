import { describe, it, expect } from 'vitest';
import { PyTorchGenerator } from '../../src/mmdnn/pytorch/generator.js';
import { PyTorchSerializer } from '../../src/mmdnn/pytorch/serializer.js';
import { Graph, Node, ValueInfo, Tensor, Attribute, Shape } from '@onnx9000/core';

function createGraph(name: string = 'TestModel') {
  return new Graph(name);
}

describe('MMDNN - PyTorch Code Generation', () => {
  describe('Generator', () => {
    it('should sanitize names correctly', () => {
      const graph = createGraph();
      graph.inputs = [new ValueInfo('1_invalid-name.v!', [1], 'float32')];
      graph.outputs = [new ValueInfo('valid_name2', [1], 'float32')];
      graph.nodes = [new Node('Relu', ['1_invalid-name.v!'], ['valid_name2'])];
      const code = new PyTorchGenerator(graph).generate();
      expect(code).toContain('v_1_invalid_name_v_');
      expect(code).toContain('valid_name2 = F.relu(v_1_invalid_name_v_)');
    });

    it('should handle getShape correctly', () => {
      const graph = createGraph();
      graph.inputs = [new ValueInfo('inp', [2, 3], 'float32')];
      graph.valueInfo = [new ValueInfo('val', [4, 5], 'float32')];
      graph.tensors['ten'] = new Tensor('ten', [6, 7], 'float32');

      const generator = new PyTorchGenerator(graph);
      expect((generator as any).getShape('inp')).toEqual([2, 3]);
      expect((generator as any).getShape('val')).toEqual([4, 5]);
      expect((generator as any).getShape('ten')).toEqual([6, 7]);
      expect((generator as any).getShape('unknown')).toBeNull();
    });

    it('should formatTuple correctly', () => {
      const graph = createGraph();
      const generator = new PyTorchGenerator(graph);
      expect((generator as any).formatTuple([1])).toEqual('(1,)');
      expect((generator as any).formatTuple([1, 2, 3])).toEqual('(1, 2, 3)');
    });

    it('should generate Conv node', () => {
      const graph = createGraph();
      graph.inputs = [new ValueInfo('x', [1, 3, 224, 224], 'float32')];
      graph.tensors['w'] = new Tensor('w', [64, 3, 3, 3], 'float32');
      // No bias
      const conv = new Node('Conv', ['x', 'w'], ['y']);
      conv.attributes = {
        group: new Attribute('group', 'INT', 1),
        strides: new Attribute('strides', 'INTS', [2, 2]),
        pads: new Attribute('pads', 'INTS', [1, 1, 1, 1]),
      };
      graph.nodes = [conv];
      graph.outputs = [new ValueInfo('y', [1, 64, 112, 112], 'float32')];
      const code = new PyTorchGenerator(graph).generate();
      expect(code).toContain(
        'nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, bias=False)',
      );
    });

    it('should generate Gemm and MatMul nodes', () => {
      const graph = createGraph();
      graph.inputs = [new ValueInfo('x', [1, 128], 'float32')];
      graph.tensors['w'] = new Tensor('w', [128, 64], 'float32');
      graph.tensors['b'] = new Tensor('b', [64], 'float32');
      const gemm = new Node('Gemm', ['x', 'w', 'b'], ['y']);
      gemm.attributes = { transB: new Attribute('transB', 'INT', 0) };
      const matmul = new Node('MatMul', ['y', 'w'], ['z']);
      graph.nodes = [gemm, matmul];
      const code = new PyTorchGenerator(graph).generate();
      expect(code).toContain('nn.Linear(in_features=128, out_features=64, bias=True)'); // Gemm
      expect(code).toContain('nn.Linear(in_features=128, out_features=64, bias=False)'); // MatMul
    });

    it('should handle Gemm with transB = 1', () => {
      const graph = createGraph();
      graph.inputs = [new ValueInfo('x', [1, 128], 'float32')];
      graph.tensors['w'] = new Tensor('w', [64, 128], 'float32'); // transposed
      const gemm = new Node('Gemm', ['x', 'w'], ['y']);
      gemm.attributes = { transB: new Attribute('transB', 'INT', 1) };
      graph.nodes = [gemm];
      const code = new PyTorchGenerator(graph).generate();
      expect(code).toContain('nn.Linear(in_features=128, out_features=64, bias=False)');
    });

    it('should generate MaxPool and AveragePool', () => {
      const graph = createGraph();
      graph.inputs = [new ValueInfo('x', [1, 3, 224, 224], 'float32')];
      const maxpool = new Node('MaxPool', ['x'], ['y']);
      maxpool.attributes = {
        kernel_shape: new Attribute('kernel_shape', 'INTS', [2, 2]),
        strides: new Attribute('strides', 'INTS', [2, 2]),
        pads: new Attribute('pads', 'INTS', [0, 0, 0, 0]),
      };
      const avgpool = new Node('AveragePool', ['y'], ['z']); // fallbacks to kernel=[2,2], stride=1, pad=0 if missing
      graph.nodes = [maxpool, avgpool];
      const code = new PyTorchGenerator(graph).generate();
      expect(code).toContain('nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))');
      expect(code).toContain('nn.AvgPool2d(kernel_size=(2, 2), stride=(1, 1), padding=(0, 0))');
    });

    it('should generate BatchNormalization', () => {
      const graph = createGraph();
      graph.inputs = [new ValueInfo('x', [1, 64, 56, 56], 'float32')];
      graph.tensors['scale'] = new Tensor('scale', [64], 'float32');
      const bn = new Node('BatchNormalization', ['x', 'scale', 'B', 'mean', 'var'], ['y']);
      bn.attributes = {
        epsilon: new Attribute('epsilon', 'FLOAT', 1e-4),
        momentum: new Attribute('momentum', 'FLOAT', 0.9),
      };
      graph.nodes = [bn];
      const code = new PyTorchGenerator(graph).generate();
      expect(code).toContain(
        'nn.BatchNorm2d(num_features=64, eps=0.0001, momentum=0.09999999999999998)',
      ); // 1.0 - 0.9 = 0.1 ish
    });

    it('should format BatchNormalization with defaults', () => {
      const graph = createGraph();
      graph.inputs = [new ValueInfo('x', [1, 64, 56, 56], 'float32')];
      graph.tensors['scale'] = new Tensor('scale', [64], 'float32');
      const bn = new Node('BatchNormalization', ['x', 'scale'], ['y']);
      graph.nodes = [bn];
      const code = new PyTorchGenerator(graph).generate();
      expect(code).toContain('nn.BatchNorm2d(num_features=64, eps=0.00001, momentum=0.1)');
    });

    it('should generate Activations (Relu, Sigmoid, Tanh) inside and outside sequences', () => {
      const graph = createGraph();
      graph.inputs = [new ValueInfo('x', [1], 'float32')];
      // seq1
      const relu = new Node('Relu', ['x'], ['r']);
      const sigm = new Node('Sigmoid', ['r'], ['s']);
      const tanh = new Node('Tanh', ['s'], ['t']);

      // standalone / branched
      const relu2 = new Node('Relu', ['x'], ['r2']);
      const sigm2 = new Node('Sigmoid', ['x'], ['s2']);
      const tanh2 = new Node('Tanh', ['x'], ['t2']);

      graph.nodes = [relu, sigm, tanh, relu2, sigm2, tanh2];
      const code = new PyTorchGenerator(graph).generate();

      // the first three should be a sequence
      expect(code).toContain('nn.Sequential(');
      expect(code).toContain('nn.ReLU(),');
      expect(code).toContain('nn.Sigmoid(),');
      expect(code).toContain('nn.Tanh(),');
      expect(code).toContain('t = self.seq_1(x)');

      // The standalone ones will fallback to F.relu, torch.sigmoid, torch.tanh
      expect(code).toContain('r2 = F.relu(x)');
      expect(code).toContain('s2 = torch.sigmoid(x)');
      expect(code).toContain('t2 = torch.tanh(x)');
    });

    it('should handle scalar math nodes (Add, Mul) and Concat', () => {
      const graph = createGraph();
      graph.inputs = [new ValueInfo('a', [1], 'float32'), new ValueInfo('b', [1], 'float32')];
      const add = new Node('Add', ['a', 'b'], ['c']);
      const mul = new Node('Mul', ['a', 'b'], ['d']);
      const concat = new Node('Concat', ['c', 'd'], ['e']);
      concat.attributes = { axis: new Attribute('axis', 'INT', 1) };
      // concat default axis
      const concat2 = new Node('Concat', ['c', 'd'], ['f']);

      graph.nodes = [add, mul, concat, concat2];
      const code = new PyTorchGenerator(graph).generate();
      expect(code).toContain('c = a + b');
      expect(code).toContain('d = a * b');
      expect(code).toContain('e = torch.cat((c, d), dim=1)');
      expect(code).toContain('f = torch.cat((c, d), dim=0)');
    });

    it('should handle Reshape with inline shape', () => {
      const graph = createGraph();
      graph.inputs = [new ValueInfo('x', [2, 3], 'float32')];

      const shapeData = new Int32Array([6]);
      const shapeTensor = new Tensor('shape', [1], 'int64');
      shapeTensor.data = shapeData;
      graph.tensors['shape'] = shapeTensor;

      const reshape = new Node('Reshape', ['x', 'shape'], ['y']);
      graph.nodes = [reshape];

      const code = new PyTorchGenerator(graph).generate();
      expect(code).toContain('y = torch.reshape(x, (6,))');
    });

    it('should handle Reshape with dynamic shape', () => {
      const graph = createGraph();
      graph.inputs = [new ValueInfo('x', [2, 3], 'float32'), new ValueInfo('shape', [2], 'int64')];
      const reshape = new Node('Reshape', ['x', 'shape'], ['y']);
      graph.nodes = [reshape];
      const code = new PyTorchGenerator(graph).generate();
      expect(code).toContain('y = torch.reshape(x, tuple(shape.tolist()))');
    });

    it('should handle Transpose', () => {
      const graph = createGraph();
      graph.inputs = [new ValueInfo('x', [1, 2, 3], 'float32')];
      const trans1 = new Node('Transpose', ['x'], ['y']);
      trans1.attributes = { perm: new Attribute('perm', 'INTS', [0, 2, 1]) };

      const trans2 = new Node('Transpose', ['x'], ['z']); // fallback

      graph.nodes = [trans1, trans2];
      const code = new PyTorchGenerator(graph).generate();
      expect(code).toContain('y = x.permute(0, 2, 1)');
      expect(code).toContain('z = torch.transpose(x, 0, 1)');
    });

    it('should handle fallback for unknown op', () => {
      const graph = createGraph();
      graph.inputs = [new ValueInfo('x', [1], 'float32')];
      const unknown = new Node('UnknownOp', ['x'], ['y']);
      graph.nodes = [unknown];
      const code = new PyTorchGenerator(graph).generate();
      expect(code).toContain("y = getattr(torch, 'unknownop')(x) # WARNING: Unmapped op");
    });

    it('should register buffers for standalone initializers in non-Sequential nodes', () => {
      const graph = createGraph();
      graph.inputs = [new ValueInfo('x', [1], 'float32')];
      graph.tensors['standalone_init'] = new Tensor('standalone_init', [1], 'float32');
      const add = new Node('Add', ['x', 'standalone_init'], ['y']);
      graph.nodes = [add];
      const code = new PyTorchGenerator(graph).generate();
      expect(code).toContain("self.register_buffer('standalone_init', torch.empty((1,)))");
      expect(code).toContain('y = x + self.standalone_init');
    });

    it('should handle a missing node in the nodes array gracefully', () => {
      const graph = createGraph();
      graph.nodes = [undefined as any]; // Force undefined
      const code = new PyTorchGenerator(graph).generate();
      expect(code).toContain('pass'); // No error, just empty forward
    });

    it('should break sequences if prevNode is missing or branching', () => {
      const graph = createGraph();
      graph.inputs = [new ValueInfo('x', [1], 'float32')];
      const relu1 = new Node('Relu', ['x'], ['r1']);
      const relu2 = new Node('Relu', ['r1'], ['r2']);
      const relu3 = new Node('Relu', ['r1'], ['r3']); // breaks sequence because r1 is used twice

      graph.nodes = [relu1, relu2, relu3];
      const code = new PyTorchGenerator(graph).generate();
      expect(code).toContain('r1 = F.relu(x)'); // standalone because of branch
      expect(code).toContain('r2 = F.relu(r1)');
      expect(code).toContain('r3 = F.relu(r1)');
    });

    it('should add prevNode to currentSeq and continue if prevNode has undefined outputs', () => {
      const graph = createGraph();
      graph.inputs = [new ValueInfo('x', [1], 'float32')];
      const relu1 = new Node('Relu', ['x'], [undefined as any]);
      const relu2 = new Node('Relu', ['x'], ['r2']);
      graph.nodes = [relu1, relu2];
      const code = new PyTorchGenerator(graph).generate();
      expect(code).toContain('F.relu(x)');
    });
  });

  describe('Serializer', () => {
    it('should correctly map dtypes to storage classes', () => {
      expect(PyTorchSerializer.getStorageClass('float32')).toBe('FloatStorage');
      expect(PyTorchSerializer.getStorageClass('float64')).toBe('DoubleStorage');
      expect(PyTorchSerializer.getStorageClass('float16')).toBe('HalfStorage');
      expect(PyTorchSerializer.getStorageClass('bfloat16')).toBe('BFloat16Storage');
      expect(PyTorchSerializer.getStorageClass('int32')).toBe('IntStorage');
      expect(PyTorchSerializer.getStorageClass('int64')).toBe('LongStorage');
      expect(PyTorchSerializer.getStorageClass('int16')).toBe('ShortStorage');
      expect(PyTorchSerializer.getStorageClass('int8')).toBe('CharStorage');
      expect(PyTorchSerializer.getStorageClass('uint8')).toBe('ByteStorage');
      expect(PyTorchSerializer.getStorageClass('bool')).toBe('BoolStorage');
      expect(PyTorchSerializer.getStorageClass('unknown' as any)).toBe('FloatStorage');
    });

    it('should serialize empty tensors array', () => {
      const bytes = PyTorchSerializer.serialize([]);
      expect(bytes).toBeDefined();
    });

    it('should skip undefined tensors gracefully', () => {
      const bytes = PyTorchSerializer.serialize([undefined as any]);
      expect(bytes).toBeDefined();
    });

    it('should serialize tensors with data', () => {
      const t1 = new Tensor('t1', [2, 2], 'float32', false, true, new Float32Array([1, 2, 3, 4]));
      const bytes = PyTorchSerializer.serialize([t1]);
      expect(bytes).toBeDefined();
    });

    it('should handle tensors with empty shape', () => {
      const t = new Tensor('empty_shape', [], 'float32', false, true, new Float32Array([1]));
      const bytes = PyTorchSerializer.serialize([t]);
      expect(bytes).toBeDefined();
    });

    it('should handle t.requiresGrad = false implicitly', () => {
      const t = new Tensor('no_grad', [1], 'float32', false, false, new Float32Array([1]));
      const bytes = PyTorchSerializer.serialize([t]);
      expect(bytes).toBeDefined();
    });

    it('should write long strings, large ints and empty data without breaking', () => {
      const name = 'A'.repeat(300); // Trigger string size > 255
      const t = new Tensor(name, [2], 'float32', false, false, null); // No data, fallback bpe=4
      const bytes = PyTorchSerializer.serialize([t]);
      expect(bytes).toBeDefined();
    });

    it('should handle int64 and float64 empty data', () => {
      const t1 = new Tensor('t1', [2], 'int64', false, false, null);
      const t2 = new Tensor('t2', [2], 'float64', false, false, null);
      const bytes = PyTorchSerializer.serialize([t1, t2]);
      expect(bytes).toBeDefined();
    });

    it('should handle float16, bfloat16, int16 empty data', () => {
      const t1 = new Tensor('t1', [2], 'float16', false, false, null);
      const t2 = new Tensor('t2', [2], 'bfloat16', false, false, null);
      const t3 = new Tensor('t3', [2], 'int16', false, false, null);
      const bytes = PyTorchSerializer.serialize([t1, t2, t3]);
      expect(bytes).toBeDefined();
    });

    it('should handle int8, uint8, bool empty data', () => {
      const t1 = new Tensor('t1', [2], 'int8', false, false, null);
      const t2 = new Tensor('t2', [2], 'uint8', false, false, null);
      const t3 = new Tensor('t3', [2], 'bool', false, false, null);
      const bytes = PyTorchSerializer.serialize([t1, t2, t3]);
      expect(bytes).toBeDefined();
    });

    it('should trigger various sizes of PickleBuilder writeInt', () => {
      // Create sizes to hit 0-255, 256-65535, 65536-2147483647, and > 2147483647
      const t1 = new Tensor('t1', [10], 'float32', false, false, null); // size 10 -> <= 255
      const t2 = new Tensor('t2', [1000], 'float32', false, false, null); // size 1000 -> <= 65535
      const t3 = new Tensor('t3', [100000], 'float32', false, false, null); // size 100000 -> <= 2147483647

      const bigTensor = new Tensor('big', [2], 'float32', false, false, new Float32Array([1, 2]));
      Object.defineProperty(bigTensor, 'size', { value: 3000000000 });

      const bytes = PyTorchSerializer.serialize([t1, t2, t3, bigTensor]);
      expect(bytes).toBeDefined();
    });
  });
});
