import os

symbolic_code = '''"""Symbolic mathematical evaluation module for ONNX shapes."""

from typing import Union, Any
import ast
import operator as op

from onnx9000.core.exceptions import ShapeInferenceError
from onnx9000.core.ir import DynamicDim

# Supported operators
operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}

def eval_expr(node: Any, context: dict[str, int]) -> Union[int, float, str]:
    if isinstance(node, ast.Num): # <number>
        return node.n
    elif isinstance(node, ast.Name):
        if node.id in context:
            return context[node.id]
        return node.id
    elif isinstance(node, ast.BinOp):
        left = eval_expr(node.left, context)
        right = eval_expr(node.right, context)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return operators[type(node.op)](left, right)
        return f"({left} {type(node.op).__name__} {right})"
    elif isinstance(node, ast.UnaryOp):
        operand = eval_expr(node.operand, context)
        if isinstance(operand, (int, float)):
            return operators[type(node.op)](operand)
        return f"(-{operand})"
    elif isinstance(node, ast.Constant):
        return node.value
    else:
        raise TypeError(node)

def evaluate_symbolic_expression(expr: str, context: dict[str, int]) -> Union[int, str]:
    """
    Evaluates a symbolic string expression given a context mapping variables to integers.
    Fallback to string if unresolved.
    """
    if expr in context:
        return context[expr]
    try:
        node = ast.parse(expr, mode='eval').body
        res = eval_expr(node, context)
        if isinstance(res, float) and res.is_integer():
            return int(res)
        return res
    except Exception:
        return expr

def broadcast_shapes(
    shape_a: tuple[Union[int, DynamicDim], ...], shape_b: tuple[Union[int, DynamicDim], ...]
) -> tuple[Union[int, DynamicDim], ...]:
    """
    Applies standard NumPy broadcasting rules to two shapes, including dynamic dimensions.
    """
    max_len = max(len(shape_a), len(shape_b))
    padded_a = (1,) * (max_len - len(shape_a)) + shape_a
    padded_b = (1,) * (max_len - len(shape_b)) + shape_b
    result = []
    for a, b in zip(padded_a, padded_b):
        a_val = a.value if isinstance(a, DynamicDim) else a
        b_val = b.value if isinstance(b, DynamicDim) else b
        
        # If one of them is 1, broadcast to the other
        if a_val == 1:
            result.append(b)
        elif b_val == 1:
            result.append(a)
        # If they are exactly equal, take either
        elif a_val == b_val:
            result.append(a)
        # Handle string symbols
        elif isinstance(a_val, str) or isinstance(b_val, str):
            result.append(DynamicDim(f"max({a_val}, {b_val})"))
        # Handle unknown dimensions (-1)
        elif a_val == -1:
            result.append(b)
        elif b_val == -1:
            result.append(a)
        else:
            raise ShapeInferenceError(
                f"Operands could not be broadcast together with shapes {shape_a} {shape_b}"
            )
    return tuple(result)

def simplify_dim(dim: Union[int, DynamicDim, str]) -> Union[int, str]:
    if isinstance(dim, DynamicDim):
        return dim.value
    return dim
'''

with open("packages/python/onnx9000-core/src/onnx9000/core/symbolic.py", "w") as f:
    f.write(symbolic_code)

shape_inference_code = '''"""Static shape inference module."""

from typing import Union, Any, Optional

from onnx9000.core.exceptions import ShapeInferenceError
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import DynamicDim, Graph, Tensor, ValueInfo, Node
from onnx9000.core.utils import topological_sort
from onnx9000.core.symbolic import broadcast_shapes, simplify_dim, evaluate_symbolic_expression

def _promote_types(t1: DType, t2: DType) -> DType:
    # A simple type promotion rule implementation
    if t1 == t2:
        return t1
    if t1 == DType.FLOAT64 or t2 == DType.FLOAT64:
        return DType.FLOAT64
    if t1 == DType.FLOAT32 or t2 == DType.FLOAT32:
        return DType.FLOAT32
    if t1 == DType.FLOAT16 or t2 == DType.FLOAT16:
        return DType.FLOAT16
    if t1 == DType.INT64 or t2 == DType.INT64:
        return DType.INT64
    if t1 == DType.INT32 or t2 == DType.INT32:
        return DType.INT32
    return t1

def get_attr(node: Node, name: str, default: Any = None) -> Any:
    for attr in node.attributes.values():
        if attr.name == name:
            return attr.value
    return default

def infer_shapes_and_types(graph: Graph) -> None:
    """
    Performs static shape and type inference on the given graph.
    Updates the graph.tensors and graph.value_info intrinsically.
    """
    try:
        sorted_nodes = topological_sort(graph)
    except Exception as e:
        raise ShapeInferenceError(f"Cannot infer shapes on a cyclic graph: {e}") from e
    
    env: dict[str, ValueInfo] = {}
    
    for inp in graph.inputs:
        if isinstance(inp, str) and inp in graph.tensors:
            t = graph.tensors[inp]
            env[inp] = ValueInfo(t.name, t.shape, t.dtype)
        elif hasattr(inp, "name"):
            env[inp.name] = inp
    for _tensor_name, tensor in graph.tensors.items():
        if tensor.is_initializer:
            env[tensor.name] = ValueInfo(tensor.name, tensor.shape, tensor.dtype)
            
    for node in sorted_nodes:
        out_shapes = []
        out_dtypes = []
        
        if node.op_type in ["Add", "Sub", "Mul", "Div", "And", "Or", "Equal", "Less", "Greater", "Where"]:
            if len(node.inputs) < 2:
                continue
            in1 = env.get(node.inputs[0])
            in2 = env.get(node.inputs[1])
            if not in1 or not in2:
                continue
            try:
                out_shape = broadcast_shapes(in1.shape, in2.shape)
            except ShapeInferenceError as e:
                raise ShapeInferenceError(f"Node {node.name} ({node.op_type}): {e}") from e
                
            out_dtype = _promote_types(in1.dtype, in2.dtype)
            if node.op_type in ["And", "Or", "Equal", "Less", "Greater"]:
                out_dtype = DType.BOOL
                
            if node.op_type == "Where" and len(node.inputs) == 3:
                in3 = env.get(node.inputs[2])
                if in3:
                    out_shape = broadcast_shapes(out_shape, in3.shape)
                    out_dtype = _promote_types(in2.dtype, in3.dtype)
            
            out_shapes = [out_shape] * len(node.outputs)
            out_dtypes = [out_dtype] * len(node.outputs)
            
        elif node.op_type in ["MatMul", "Gemm"]:
            if len(node.inputs) < 2:
                continue
            in1 = env.get(node.inputs[0])
            in2 = env.get(node.inputs[1])
            if not in1 or not in2:
                continue
            shape1 = list(in1.shape)
            shape2 = list(in2.shape)
            
            transA = get_attr(node, "transA", 0) if node.op_type == "Gemm" else 0
            transB = get_attr(node, "transB", 0) if node.op_type == "Gemm" else 0
            
            if transA and len(shape1) >= 2:
                shape1[-1], shape1[-2] = shape1[-2], shape1[-1]
            if transB and len(shape2) >= 2:
                shape2[-1], shape2[-2] = shape2[-2], shape2[-1]
                
            if len(shape1) >= 2 and len(shape2) >= 2:
                batch_shape = []
                if len(shape1) > 2 or len(shape2) > 2:
                    batch1 = tuple(shape1[:-2])
                    batch2 = tuple(shape2[:-2])
                    try:
                        batch_shape = list(broadcast_shapes(batch1, batch2))
                    except ShapeInferenceError:
                        batch_shape = list(batch1) if len(batch1) > len(batch2) else list(batch2)
                
                out_shape = tuple(batch_shape + [shape1[-2], shape2[-1]])
            else:
                out_shape = tuple() 
                
            out_dtype = _promote_types(in1.dtype, in2.dtype)
            out_shapes = [out_shape] * len(node.outputs)
            out_dtypes = [out_dtype] * len(node.outputs)
            
        elif node.op_type in ["Relu", "Sigmoid", "Tanh", "Exp", "Log", "Cast", "Shape", "Size"]:
            if len(node.inputs) < 1:
                continue
            in1 = env.get(node.inputs[0])
            if not in1:
                continue
            
            if node.op_type == "Cast":
                to_type = get_attr(node, "to", DType.FLOAT32.value)
                out_dtype = DType(to_type)
                out_shape = in1.shape
            elif node.op_type == "Shape":
                out_dtype = DType.INT64
                out_shape = (len(in1.shape),)
            elif node.op_type == "Size":
                out_dtype = DType.INT64
                out_shape = ()
            else:
                out_dtype = in1.dtype
                out_shape = in1.shape
                
            out_shapes = [out_shape] * len(node.outputs)
            out_dtypes = [out_dtype] * len(node.outputs)
            
        elif node.op_type == "Reshape":
            if len(node.inputs) < 2:
                continue
            in1 = env.get(node.inputs[0])
            if not in1:
                continue
            
            shape_tensor_name = node.inputs[1]
            out_shape = None
            if shape_tensor_name in graph.tensors:
                shape_tensor = graph.tensors[shape_tensor_name]
                if hasattr(shape_tensor, "data") and shape_tensor.data:
                    pass
            
            if not out_shape and hasattr(graph.tensors.get(shape_tensor_name, None), "values"):
                vals = getattr(graph.tensors[shape_tensor_name], "values", None)
                if vals is not None:
                    target_shape = list(vals)
                    try:
                        in_vol = 1
                        for d in in1.shape:
                            in_vol *= int(simplify_dim(d))
                        
                        out_vol = 1
                        neg_idx = -1
                        for i, d in enumerate(target_shape):
                            if d == -1:
                                neg_idx = i
                            else:
                                out_vol *= int(simplify_dim(d))
                        
                        if neg_idx != -1 and out_vol != 0:
                            target_shape[neg_idx] = in_vol // out_vol
                        out_shape = tuple(target_shape)
                    except Exception:
                        out_shape = tuple(vals)
            
            if not out_shape:
                s_shape = env.get(shape_tensor_name, ValueInfo("",(),DType.INT64)).shape
                dim_count = s_shape[0] if s_shape else 1
                out_shape = tuple(DynamicDim(f"dim_{i}") for i in range(dim_count if isinstance(dim_count, int) else 1))
            
            out_shapes = [out_shape] * len(node.outputs)
            out_dtypes = [in1.dtype] * len(node.outputs)

        elif node.op_type in ["Conv", "MaxPool", "AveragePool", "GlobalAveragePool"]:
            if len(node.inputs) < 1:
                continue
            in1 = env.get(node.inputs[0])
            if not in1:
                continue
            
            in_shape = list(in1.shape)
            out_dtype = in1.dtype
            
            if node.op_type == "GlobalAveragePool":
                out_shape = tuple(in_shape[:2] + [1] * (len(in_shape) - 2))
                out_shapes = [out_shape] * len(node.outputs)
                out_dtypes = [out_dtype] * len(node.outputs)
            else:
                kernel_shape = get_attr(node, "kernel_shape", [])
                strides = get_attr(node, "strides", [1] * len(kernel_shape))
                pads = get_attr(node, "pads", [0] * (2 * len(kernel_shape)))
                dilations = get_attr(node, "dilations", [1] * len(kernel_shape))
                
                if node.op_type == "Conv" and len(node.inputs) > 1:
                    w_info = env.get(node.inputs[1])
                    if w_info and len(w_info.shape) > 2:
                        kernel_shape = list(w_info.shape[2:])
                        out_channels = w_info.shape[0]
                    else:
                        out_channels = DynamicDim("C_out")
                else:
                    out_channels = in_shape[1] if len(in_shape) > 1 else DynamicDim("C_out")

                spatial_dims = []
                for i in range(len(kernel_shape)):
                    try:
                        in_dim = in_shape[2 + i]
                        if isinstance(in_dim, DynamicDim) or isinstance(in_dim, str):
                            spatial_dims.append(DynamicDim(f"spatial_{i}"))
                        else:
                            k = kernel_shape[i]
                            s = strides[i]
                            p = pads[i] + pads[i + len(kernel_shape)]
                            d = dilations[i]
                            out_dim = (in_dim + p - ((k - 1) * d + 1)) // s + 1
                            spatial_dims.append(out_dim)
                    except Exception:
                        spatial_dims.append(DynamicDim(f"spatial_{i}"))
                
                out_shape = tuple([in_shape[0] if len(in_shape) > 0 else 1, out_channels] + spatial_dims)
                out_shapes = [out_shape] * len(node.outputs)
                out_dtypes = [out_dtype] * len(node.outputs)

        for i, out_name in enumerate(node.outputs):
            if i < len(out_shapes):
                s = out_shapes[i]
                d = out_dtypes[i]
            else:
                in0 = env.get(node.inputs[0]) if node.inputs else None
                s = ()
                d = in0.dtype if in0 else DType.FLOAT32
                
            env[out_name] = ValueInfo(out_name, s, d)
            if out_name not in graph.tensors:
                graph.add_tensor(Tensor(out_name, s, d))
            else:
                graph.tensors[out_name].shape = s
                graph.tensors[out_name].dtype = d

'''

with open("packages/python/onnx9000-core/src/onnx9000/core/shape_inference.py", "w") as f:
    f.write(shape_inference_code)

print("Created symbolic.py and shape_inference.py")
