import { describe, it, expect } from 'vitest';
import { cleanName, ONNXToPyTorchVisitor } from '../src/codegen/pytorch.js';
import { Graph } from '../src/ir/graph.js';
import { Node } from '../src/ir/node.js';

describe('cleanName', () => {
  it('should replace non-alphanumeric chars with _', () => {
    expect(cleanName('a.b-c')).toBe('a_b_c');
  });
  it('should return var_empty for empty string', () => {
    expect(cleanName('')).toBe('var_empty');
  });
});

describe('ONNXToPyTorchVisitor', () => {
  it('should generate empty module for empty graph', () => {
    const g = new Graph('test');
    const visitor = new ONNXToPyTorchVisitor(g);
    const code = visitor.generate();
    expect(code).toContain('class GeneratedModel(nn.Module):');
    expect(code).toContain('def forward(self):');
    expect(code).toContain('pass');
    expect(code).toContain('return None');
  });

  it('should register buffers for initializers', () => {
    const g = new Graph('test');
    g.tensors['my_weight'] = { isInitializer: true, shape: [2, 2] } as any;
    g.tensors['my_other'] = { isInitializer: false, shape: [2] } as any;
    const visitor = new ONNXToPyTorchVisitor(g);
    const code = visitor.generate();
    expect(code).toContain("self.register_buffer('my_weight', torch.zeros(2, 2))");
    expect(code).not.toContain('my_other');
  });

  it('should generate known ops correctly', () => {
    const g = new Graph('test');
    g.inputs.push({ name: 'input1', type: null as any });
    g.outputs.push({ name: 'output1', type: null as any });

    const node = new Node('Relu');
    node.inputs = ['input1'];
    node.outputs = ['output1'];
    g.nodes.push(node);

    const visitor = new ONNXToPyTorchVisitor(g);
    const code = visitor.generate();
    expect(code).toContain('output1 = F.relu(input1)');
    expect(code).toContain('return output1');
  });

  it('should generate Conv, MatMul, Add, Reshape', () => {
    const g = new Graph('test');

    const n1 = new Node('Conv');
    n1.inputs = ['A', 'W'];
    n1.outputs = ['C1'];
    const n2 = new Node('MatMul');
    n2.inputs = ['C1', 'B'];
    n2.outputs = ['C2'];
    const n3 = new Node('Add');
    n3.inputs = ['C2', 'bias'];
    n3.outputs = ['C3'];
    const n4 = new Node('Reshape');
    n4.inputs = ['C3', 'shape'];
    n4.outputs = ['out'];

    g.nodes.push(n1, n2, n3, n4);

    const visitor = new ONNXToPyTorchVisitor(g);
    const code = visitor.generate();
    expect(code).toContain('C1 = F.conv2d(A, self.W)');
    expect(code).toContain('C2 = torch.matmul(C1, B)');
    expect(code).toContain('C3 = C2 + bias');
    expect(code).toContain('out = torch.reshape(C3, self.shape.tolist())');
  });

  it('should generate unknown ops dynamically', () => {
    const g = new Graph('test');
    const node = new Node('UnknownOp');
    node.inputs = ['A', 'B'];
    node.outputs = ['C'];
    g.nodes.push(node);
    const visitor = new ONNXToPyTorchVisitor(g);
    const code = visitor.generate();
    expect(code).toContain('C = getattr(torch.ops.onnx, "unknownop")(A, B)');
  });
});
