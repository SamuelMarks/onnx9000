from typing import Any, Dict, List, Optional

import onnx
from onnx.helper import tensor_dtype_to_np_dtype

from ...relay.expr import Call, Constant, Expr, Op, TupleExpr, TupleGetItem, Var
from ...relay.module import IRModule
from ...relay.ty import TensorType, TupleType


class ONNXImporter:
    """Importer for ONNX models to WebRelay IR."""

    def __init__(self):
        self._nodes: dict[str, Expr] = {}
        # Op converters
        self._convert_map: dict[str, callable] = {
            "Add": self._convert_add,
            "Sub": self._convert_sub,
            "Mul": self._convert_mul,
            "Div": self._convert_div,
            "MatMul": self._convert_matmul,
            "Gemm": self._convert_gemm,
            "Conv": self._convert_conv,
            "Relu": self._convert_relu,
            "LeakyRelu": self._convert_leakyrelu,
            "Sigmoid": self._convert_sigmoid,
            "Tanh": self._convert_tanh,
            "Softmax": self._convert_softmax,
            "LogSoftmax": self._convert_logsoftmax,
            "Erf": self._convert_erf,
            "Gelu": self._convert_gelu,
            "MaxPool": self._convert_maxpool,
            "AveragePool": self._convert_averagepool,
            "GlobalMaxPool": self._convert_globalmaxpool,
            "GlobalAveragePool": self._convert_globalaveragepool,
            "Pad": self._convert_pad,
            "Reshape": self._convert_reshape,
            "Flatten": self._convert_flatten,
            "Transpose": self._convert_transpose,
            "Squeeze": self._convert_squeeze,
            "Unsqueeze": self._convert_unsqueeze,
            "Concat": self._convert_concat,
            "Split": self._convert_split,
            "Slice": self._convert_slice,
            "Gather": self._convert_gather,
            "GatherElements": self._convert_gatherelements,
            "GatherND": self._convert_gathernd,
            "Scatter": self._convert_scatter,
            "ScatterElements": self._convert_scatterelements,
            "ScatterND": self._convert_scatternd,
            "Cast": self._convert_cast,
            "ReduceSum": self._convert_reducesum,
            "ReduceMean": self._convert_reducemean,
            "ReduceMax": self._convert_reducemax,
            "ReduceMin": self._convert_reducemin,
            "ReduceProd": self._convert_reduceprod,
            "ArgMax": self._convert_argmax,
            "ArgMin": self._convert_argmin,
            "Shape": self._convert_shape,
            "Size": self._convert_size,
            "ConstantOfShape": self._convert_constantofshape,
            "Expand": self._convert_expand,
            "Tile": self._convert_tile,
            "Where": self._convert_where,
            "Less": self._convert_less,
            "LessOrEqual": self._convert_lessorequal,
            "Greater": self._convert_greater,
            "GreaterOrEqual": self._convert_greaterorequal,
            "Equal": self._convert_equal,
            "Not": self._convert_not,
            "And": self._convert_and,
            "Or": self._convert_or,
            "Xor": self._convert_xor,
            "IsNaN": self._convert_isnan,
            "IsInf": self._convert_isinf,
            "Sign": self._convert_sign,
            "Abs": self._convert_abs,
            "Neg": self._convert_neg,
            "Ceil": self._convert_ceil,
            "Floor": self._convert_floor,
            "Round": self._convert_round,
            "Sqrt": self._convert_sqrt,
            "Pow": self._convert_pow,
            "Exp": self._convert_exp,
            "Log": self._convert_log,
            "Sin": self._convert_sin,
            "Cos": self._convert_cos,
            "Tan": self._convert_tan,
            "Asin": self._convert_asin,
            "Acos": self._convert_acos,
            "Atan": self._convert_atan,
            "Sinh": self._convert_sinh,
            "Cosh": self._convert_cosh,
            "Asinh": self._convert_asinh,
            "Acosh": self._convert_acosh,
            "Atanh": self._convert_atanh,
            "Clip": self._convert_clip,
            "BatchNormalization": self._convert_batchnorm,
            "InstanceNormalization": self._convert_instancenorm,
            "LayerNormalization": self._convert_layernorm,
            "Dropout": self._convert_dropout,
            "RNN": self._convert_rnn,
            "LSTM": self._convert_lstm,
            "GRU": self._convert_gru,
            "TopK": self._convert_topk,
            "NonZero": self._convert_nonzero,
            "Resize": self._convert_resize,
            "OneHot": self._convert_onehot,
            "CumSum": self._convert_cumsum,
        }

    def _get_type(self, tensor_type) -> TensorType:
        """Map ONNX tensor type to IR TensorType"""
        dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type).name
        shape = []
        for d in tensor_type.shape.dim:
            if d.HasField("dim_value"):
                shape.append(d.dim_value)
            elif d.HasField("dim_param"):
                shape.append(d.dim_param)
            else:
                shape.append("?")
        return TensorType(shape=tuple(shape), dtype=dtype)

    def _parse_attr(self, attr: onnx.AttributeProto) -> Any:
        if attr.type == onnx.AttributeProto.FLOAT:
            return attr.f
        elif attr.type == onnx.AttributeProto.INT:
            return attr.i
        elif attr.type == onnx.AttributeProto.STRING:
            return attr.s.decode("utf-8")
        elif attr.type == onnx.AttributeProto.TENSOR:
            import numpy as np
            from onnx.numpy_helper import to_array

            return to_array(attr.t)
        elif attr.type == onnx.AttributeProto.FLOATS:
            return list(attr.floats)
        elif attr.type == onnx.AttributeProto.INTS:
            return list(attr.ints)
        elif attr.type == onnx.AttributeProto.STRINGS:
            return [s.decode("utf-8") for s in attr.strings]
        # Not implementing graph/sparse_tensor/etc for now
        return None

    def from_onnx(self, model: onnx.ModelProto, opset: int = None) -> IRModule:
        graph = model.graph

        # 1. Parse Inputs
        inputs = []
        for inp in graph.input:
            ty = self._get_type(inp.type.tensor_type)
            var = Var(name_hint=inp.name, type_annotation=ty)
            self._nodes[inp.name] = var
            inputs.append(var)

        # 2. Parse Initializers
        import numpy as np
        from onnx.numpy_helper import to_array

        for init in graph.initializer:
            arr = to_array(init)
            ty = TensorType(shape=tuple(arr.shape), dtype=str(arr.dtype))
            const = Constant(data=arr, type_annotation=ty)
            self._nodes[init.name] = const

        # 3. Parse Nodes
        for node in graph.node:
            op_name = node.op_type

            # Extract inputs
            node_inputs = [self._nodes.get(inp, None) for inp in node.input if inp]

            # Extract attrs
            attrs = {attr.name: self._parse_attr(attr) for attr in node.attribute}

            # Convert
            if op_name in self._convert_map:
                res = self._convert_map[op_name](node_inputs, attrs)
            else:
                # Fallback to generic call
                res = Call(op=Op(op_name), args=node_inputs, attrs=attrs)

            # Assign outputs
            if len(node.output) == 1:
                self._nodes[node.output[0]] = res
            else:
                for i, out_name in enumerate(node.output):
                    if out_name:
                        self._nodes[out_name] = TupleGetItem(tuple_value=res, index=i)

        # 4. Construct Outputs
        outputs = [self._nodes[out.name] for out in graph.output]

        if len(outputs) == 1:
            body = outputs[0]
        else:
            body = TupleExpr(fields=outputs)

        # 5. Create Function
        from ...relay.expr import Function

        func = Function(params=inputs, body=body)

        mod = IRModule()
        mod.add(Var("main"), func)
        return mod

    # --- Node Converters ---

    def _generic_convert(self, op_name: str, inputs: list[Expr], attrs: dict[str, Any]) -> Expr:
        return Call(op=Op(op_name), args=inputs, attrs=attrs)

    def _convert_add(self, inputs, attrs):
        return self._generic_convert("Add", inputs, attrs)

    def _convert_sub(self, inputs, attrs):
        return self._generic_convert("Sub", inputs, attrs)

    def _convert_mul(self, inputs, attrs):
        return self._generic_convert("Multiply", inputs, attrs)

    def _convert_div(self, inputs, attrs):
        return self._generic_convert("Divide", inputs, attrs)

    def _convert_matmul(self, inputs, attrs):
        return self._generic_convert("MatMul", inputs, attrs)

    def _convert_gemm(self, inputs, attrs):
        return self._generic_convert("Gemm", inputs, attrs)

    def _convert_conv(self, inputs, attrs):
        return self._generic_convert("Conv", inputs, attrs)

    def _convert_relu(self, inputs, attrs):
        return self._generic_convert("Relu", inputs, attrs)

    def _convert_leakyrelu(self, inputs, attrs):
        return self._generic_convert("LeakyRelu", inputs, attrs)

    def _convert_sigmoid(self, inputs, attrs):
        return self._generic_convert("Sigmoid", inputs, attrs)

    def _convert_tanh(self, inputs, attrs):
        return self._generic_convert("Tanh", inputs, attrs)

    def _convert_softmax(self, inputs, attrs):
        return self._generic_convert("Softmax", inputs, attrs)

    def _convert_logsoftmax(self, inputs, attrs):
        return self._generic_convert("LogSoftmax", inputs, attrs)

    def _convert_erf(self, inputs, attrs):
        return self._generic_convert("Erf", inputs, attrs)

    def _convert_gelu(self, inputs, attrs):
        return self._generic_convert("Gelu", inputs, attrs)

    def _convert_maxpool(self, inputs, attrs):
        return self._generic_convert("MaxPool", inputs, attrs)

    def _convert_averagepool(self, inputs, attrs):
        return self._generic_convert("AveragePool", inputs, attrs)

    def _convert_globalmaxpool(self, inputs, attrs):
        return self._generic_convert("GlobalMaxPool", inputs, attrs)

    def _convert_globalaveragepool(self, inputs, attrs):
        return self._generic_convert("GlobalAveragePool", inputs, attrs)

    def _convert_pad(self, inputs, attrs):
        return self._generic_convert("Pad", inputs, attrs)

    def _convert_reshape(self, inputs, attrs):
        return self._generic_convert("Reshape", inputs, attrs)

    def _convert_flatten(self, inputs, attrs):
        return self._generic_convert("Flatten", inputs, attrs)

    def _convert_transpose(self, inputs, attrs):
        return self._generic_convert("Transpose", inputs, attrs)

    def _convert_squeeze(self, inputs, attrs):
        return self._generic_convert("Squeeze", inputs, attrs)

    def _convert_unsqueeze(self, inputs, attrs):
        return self._generic_convert("Unsqueeze", inputs, attrs)

    def _convert_concat(self, inputs, attrs):
        return self._generic_convert("Concat", inputs, attrs)

    def _convert_split(self, inputs, attrs):
        return self._generic_convert("Split", inputs, attrs)

    def _convert_slice(self, inputs, attrs):
        return self._generic_convert("Slice", inputs, attrs)

    def _convert_gather(self, inputs, attrs):
        return self._generic_convert("Gather", inputs, attrs)

    def _convert_gatherelements(self, inputs, attrs):
        return self._generic_convert("GatherElements", inputs, attrs)

    def _convert_gathernd(self, inputs, attrs):
        return self._generic_convert("GatherND", inputs, attrs)

    def _convert_scatter(self, inputs, attrs):
        return self._generic_convert("Scatter", inputs, attrs)

    def _convert_scatterelements(self, inputs, attrs):
        return self._generic_convert("ScatterElements", inputs, attrs)

    def _convert_scatternd(self, inputs, attrs):
        return self._generic_convert("ScatterND", inputs, attrs)

    def _convert_cast(self, inputs, attrs):
        return self._generic_convert("Cast", inputs, attrs)

    def _convert_reducesum(self, inputs, attrs):
        return self._generic_convert("ReduceSum", inputs, attrs)

    def _convert_reducemean(self, inputs, attrs):
        return self._generic_convert("ReduceMean", inputs, attrs)

    def _convert_reducemax(self, inputs, attrs):
        return self._generic_convert("ReduceMax", inputs, attrs)

    def _convert_reducemin(self, inputs, attrs):
        return self._generic_convert("ReduceMin", inputs, attrs)

    def _convert_reduceprod(self, inputs, attrs):
        return self._generic_convert("ReduceProd", inputs, attrs)

    def _convert_argmax(self, inputs, attrs):
        return self._generic_convert("ArgMax", inputs, attrs)

    def _convert_argmin(self, inputs, attrs):
        return self._generic_convert("ArgMin", inputs, attrs)

    def _convert_shape(self, inputs, attrs):
        return self._generic_convert("Shape", inputs, attrs)

    def _convert_size(self, inputs, attrs):
        return self._generic_convert("Size", inputs, attrs)

    def _convert_constantofshape(self, inputs, attrs):
        return self._generic_convert("ConstantOfShape", inputs, attrs)

    def _convert_expand(self, inputs, attrs):
        return self._generic_convert("Expand", inputs, attrs)

    def _convert_tile(self, inputs, attrs):
        return self._generic_convert("Tile", inputs, attrs)

    def _convert_where(self, inputs, attrs):
        return self._generic_convert("Where", inputs, attrs)

    def _convert_less(self, inputs, attrs):
        return self._generic_convert("Less", inputs, attrs)

    def _convert_lessorequal(self, inputs, attrs):
        return self._generic_convert("LessOrEqual", inputs, attrs)

    def _convert_greater(self, inputs, attrs):
        return self._generic_convert("Greater", inputs, attrs)

    def _convert_greaterorequal(self, inputs, attrs):
        return self._generic_convert("GreaterOrEqual", inputs, attrs)

    def _convert_equal(self, inputs, attrs):
        return self._generic_convert("Equal", inputs, attrs)

    def _convert_not(self, inputs, attrs):
        return self._generic_convert("Not", inputs, attrs)

    def _convert_and(self, inputs, attrs):
        return self._generic_convert("And", inputs, attrs)

    def _convert_or(self, inputs, attrs):
        return self._generic_convert("Or", inputs, attrs)

    def _convert_xor(self, inputs, attrs):
        return self._generic_convert("Xor", inputs, attrs)

    def _convert_isnan(self, inputs, attrs):
        return self._generic_convert("IsNaN", inputs, attrs)

    def _convert_isinf(self, inputs, attrs):
        return self._generic_convert("IsInf", inputs, attrs)

    def _convert_sign(self, inputs, attrs):
        return self._generic_convert("Sign", inputs, attrs)

    def _convert_abs(self, inputs, attrs):
        return self._generic_convert("Abs", inputs, attrs)

    def _convert_neg(self, inputs, attrs):
        return self._generic_convert("Neg", inputs, attrs)

    def _convert_ceil(self, inputs, attrs):
        return self._generic_convert("Ceil", inputs, attrs)

    def _convert_floor(self, inputs, attrs):
        return self._generic_convert("Floor", inputs, attrs)

    def _convert_round(self, inputs, attrs):
        return self._generic_convert("Round", inputs, attrs)

    def _convert_sqrt(self, inputs, attrs):
        return self._generic_convert("Sqrt", inputs, attrs)

    def _convert_pow(self, inputs, attrs):
        return self._generic_convert("Pow", inputs, attrs)

    def _convert_exp(self, inputs, attrs):
        return self._generic_convert("Exp", inputs, attrs)

    def _convert_log(self, inputs, attrs):
        return self._generic_convert("Log", inputs, attrs)

    def _convert_sin(self, inputs, attrs):
        return self._generic_convert("Sin", inputs, attrs)

    def _convert_cos(self, inputs, attrs):
        return self._generic_convert("Cos", inputs, attrs)

    def _convert_tan(self, inputs, attrs):
        return self._generic_convert("Tan", inputs, attrs)

    def _convert_asin(self, inputs, attrs):
        return self._generic_convert("Asin", inputs, attrs)

    def _convert_acos(self, inputs, attrs):
        return self._generic_convert("Acos", inputs, attrs)

    def _convert_atan(self, inputs, attrs):
        return self._generic_convert("Atan", inputs, attrs)

    def _convert_sinh(self, inputs, attrs):
        return self._generic_convert("Sinh", inputs, attrs)

    def _convert_cosh(self, inputs, attrs):
        return self._generic_convert("Cosh", inputs, attrs)

    def _convert_asinh(self, inputs, attrs):
        return self._generic_convert("Asinh", inputs, attrs)

    def _convert_acosh(self, inputs, attrs):
        return self._generic_convert("Acosh", inputs, attrs)

    def _convert_atanh(self, inputs, attrs):
        return self._generic_convert("Atanh", inputs, attrs)

    def _convert_clip(self, inputs, attrs):
        return self._generic_convert("Clip", inputs, attrs)

    def _convert_batchnorm(self, inputs, attrs):
        return self._generic_convert("BatchNormalization", inputs, attrs)

    def _convert_instancenorm(self, inputs, attrs):
        return self._generic_convert("InstanceNormalization", inputs, attrs)

    def _convert_layernorm(self, inputs, attrs):
        return self._generic_convert("LayerNormalization", inputs, attrs)

    def _convert_dropout(self, inputs, attrs):
        return self._generic_convert("Dropout", inputs, attrs)

    def _convert_rnn(self, inputs, attrs):
        return self._generic_convert("RNN", inputs, attrs)

    def _convert_lstm(self, inputs, attrs):
        return self._generic_convert("LSTM", inputs, attrs)

    def _convert_gru(self, inputs, attrs):
        return self._generic_convert("GRU", inputs, attrs)

    def _convert_topk(self, inputs, attrs):
        return self._generic_convert("TopK", inputs, attrs)

    def _convert_nonzero(self, inputs, attrs):
        return self._generic_convert("NonZero", inputs, attrs)

    def _convert_resize(self, inputs, attrs):
        return self._generic_convert("Resize", inputs, attrs)

    def _convert_onehot(self, inputs, attrs):
        return self._generic_convert("OneHot", inputs, attrs)

    def _convert_cumsum(self, inputs, attrs):
        return self._generic_convert("CumSum", inputs, attrs)


def from_onnx(model, opset: int = None) -> IRModule:
    """Convenience function to import ONNX models."""
    return ONNXImporter().from_onnx(model, opset)
