"""Advanced diagnostic utility for profiling MACs, FLOPs, parameters, and memory."""

import sys
from typing import Any, Optional, Union

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Constant, DynamicDim, Graph, Node, Variable
from onnx9000.core.shape_inference import infer_shapes_and_types
from onnx9000.core.symbolic import evaluate_symbolic_expression, simplify_dim


def dtype_size(dtype: DType) -> int:
    mapping = {
        DType.FLOAT32: 4,
        DType.FLOAT64: 8,
        DType.INT8: 1,
        DType.INT16: 2,
        DType.INT32: 4,
        DType.INT64: 8,
        DType.UINT8: 1,
        DType.UINT16: 2,
        DType.UINT32: 4,
        DType.UINT64: 8,
        DType.BOOL: 1,
        DType.FLOAT16: 2,
        DType.BFLOAT16: 2,
    }
    return mapping.get(dtype, 4)


def resolve_volume(shape: tuple, dynamic_overrides: dict[str, int] = None) -> Union[int, str]:
    if not shape:
        return 1
    vol = 1
    symbolic_vol = []
    for d in shape:
        sd = simplify_dim(d)
        if isinstance(sd, str):
            if dynamic_overrides and sd in dynamic_overrides:
                vol *= dynamic_overrides[sd]
            else:
                symbolic_vol.append(sd)
        elif isinstance(sd, int) and sd > 0:
            vol *= sd

    if symbolic_vol:
        sym_str = " * ".join(symbolic_vol)
        return f"{vol} * {sym_str}" if vol != 1 else sym_str
    return vol


def get_attr(node: Node, name: str, default: Any = None) -> Any:
    for attr in node.attributes.values():
        if attr.name == name:
            return attr.value
    return default


import time
from functools import wraps


def profile_graph(func):
    @wraps(func)
    def wrapper(graph, *args, **kwargs):
        start = time.perf_counter()
        res = profile(graph)
        elapsed = time.perf_counter() - start
        print(f"--- Profiler Result ({elapsed * 1000:.2f}ms) ---")
        print(res)
        return func(graph, *args, **kwargs)

    return wrapper


class ProfilerResult:
    def __init__(self):
        self.total_macs: Union[int, str] = 0
        self.total_flops: Union[int, str] = 0
        self.total_params: int = 0
        self.total_memory_bytes: int = 0
        self.peak_activation_bytes: int = 0

        # Precision specific MACs
        self.fp32_macs: Union[int, str] = 0
        self.fp16_macs: Union[int, str] = 0
        self.int8_macs: Union[int, str] = 0
        self.int4_macs: Union[int, str] = 0

        self.node_profiles: list[dict] = []
        self.suggestions: list[str] = []

        self.float64_count = 0
        self.int64_count = 0

    def generate_suggestions(self):
        if self.float64_count > 0:
            self.suggestions.append(
                f"Found {self.float64_count} tensors using Float64. Consider downcasting to Float32 to save 50% memory."
            )
        if self.int64_count > 0:
            self.suggestions.append(
                f"Found {self.int64_count} tensors using Int64. Consider downcasting to Int32."
            )
        if (
            isinstance(self.total_flops, int)
            and isinstance(self.total_memory_bytes, int)
            and self.total_memory_bytes > 0
        ):
            intensity = self.total_flops / self.total_memory_bytes
            if intensity < 1.0:
                self.suggestions.append(
                    f"Model is severely memory bound. Arithmetic intensity is {intensity:.2f} FLOPs/Byte."
                )

        # Check WebGPU 128MB limits
        for n in self.node_profiles:
            params = n.get("params", 0)
            activations = n.get("activation_bytes", 0)
            if isinstance(params, int) and params > 134217728:
                self.suggestions.append(
                    f"Node {n['name']} parameter size exceeds WebGPU 128MB limit ({params / 1e6:.2f} MB)."
                )
            if isinstance(activations, int) and activations > 134217728:
                self.suggestions.append(
                    f"Node {n['name']} activation size exceeds WebGPU 128MB limit ({activations / 1e6:.2f} MB)."
                )

    def estimate_latency(self, hardware_tops: float = 1.0, hardware_bw_gbps: float = 10.0) -> dict:
        """
        Estimates the lower-bound latency based on Roofline model constraints.
        hardware_tops: TeraOps per second (e.g. Apple M1 is ~2.6 TFLOPS FP32)
        hardware_bw_gbps: Memory bandwidth in GB/s (e.g. Apple M1 is ~68 GB/s)
        """
        compute_time_ms = 0.0
        memory_time_ms = 0.0

        if isinstance(self.total_flops, int):
            compute_time_ms = (self.total_flops / (hardware_tops * 1e12)) * 1000.0

        total_mem = 0
        if isinstance(self.total_memory_bytes, int):
            total_mem += self.total_memory_bytes
        if isinstance(self.peak_activation_bytes, int):
            total_mem += self.peak_activation_bytes

        memory_time_ms = (total_mem / (hardware_bw_gbps * 1e9)) * 1000.0

        bottleneck = "Compute Bound" if compute_time_ms > memory_time_ms else "Memory Bound"
        return {
            "compute_latency_ms": round(compute_time_ms, 2),
            "memory_latency_ms": round(memory_time_ms, 2),
            "total_estimated_latency_ms": round(max(compute_time_ms, memory_time_ms), 2),
            "bottleneck": bottleneck,
        }

    def print_parameter_pie_chart(self):
        print("--- Parameter Distribution ---")
        if self.total_params == 0:
            print("No parameters found.")
            return
        # group by node type since parameters belong to nodes
        op_types = {}
        # Simple heuristic: we track params per node in node_profiles
        for n in self.node_profiles:
            if n.get("params", 0) > 0:
                op_types[n["op_type"]] = op_types.get(n["op_type"], 0) + n["params"]
        for op, count in sorted(op_types.items(), key=lambda x: x[1], reverse=True):
            pct = (count / self.total_params) * 100
            print(f"{op}: {pct:.1f}% ({count} Params)")

    def print_activation_pie_chart(self):
        print("--- Activation Distribution ---")
        if self.peak_activation_bytes == 0:
            print("No activations found.")
            return
        op_types = {}
        for n in self.node_profiles:
            if n.get("activation_bytes", 0) > 0:
                op_types[n["op_type"]] = op_types.get(n["op_type"], 0) + n["activation_bytes"]
        for op, count in sorted(op_types.items(), key=lambda x: x[1], reverse=True):
            pct = (count / self.peak_activation_bytes) * 100
            print(f"{op}: {pct:.1f}% ({count / 1e6:.2f} MB)")

    def __repr__(self):
        return (
            f"ProfilerResult(macs={self.total_macs}, flops={self.total_flops}, "
            f"params={self.total_params}, memory={self.total_memory_bytes / 1e6:.2f} MB)"
        )

    def print_bottleneck_analysis(self, top_k: int = 5):
        print(f"--- Top {top_k} Compute Bottlenecks ---")
        sorted_nodes = sorted(
            [n for n in self.node_profiles if isinstance(n["flops"], int)],
            key=lambda x: x["flops"],
            reverse=True,
        )
        for i, n in enumerate(sorted_nodes[:top_k]):
            print(
                f"{i + 1}. {n['name']} ({n['op_type']}) - FLOPs: {n['flops']} - Arithmetic Intensity: {n.get('arithmetic_intensity', 'N/A')}"
            )

    def print_distribution_pie_chart(self):
        print("--- MACs/FLOPs Distribution ---")
        op_types = {}
        for n in self.node_profiles:
            if isinstance(n["flops"], int) and n["flops"] > 0:
                op_types[n["op_type"]] = op_types.get(n["op_type"], 0) + n["flops"]

        if not op_types or not isinstance(self.total_flops, int) or self.total_flops == 0:
            print("Cannot compute pie chart: FLOPs are dynamic or zero.")
            return

        for op, count in sorted(op_types.items(), key=lambda x: x[1], reverse=True):
            pct = (count / self.total_flops) * 100
            print(f"{op}: {pct:.1f}% ({count} FLOPs)")

    def get_cumulative_flops_up_to(self, node_name: str) -> Union[int, str]:
        total = 0
        for n in self.node_profiles:
            total = _add_metric(total, n["flops"])
            if n["name"] == node_name:
                break
        return total


def _add_metric(a: Union[int, str], b: Union[int, str]) -> Union[int, str]:
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    if a == 0:
        return b
    if b == 0:
        return a
    return f"({a} + {b})"


def profile(graph: Graph, dynamic_overrides: dict[str, int] = None) -> ProfilerResult:
    """Profiles the graph for MACs, FLOPs, Parameter count, and Memory usage."""
    infer_shapes_and_types(graph)
    res = ProfilerResult()

    for t in graph.tensors.values():
        if isinstance(t, Constant) or t.is_initializer:
            vol = resolve_volume(t.shape, dynamic_overrides)
            if isinstance(vol, int) and vol > 0:
                if t.dtype == DType.FLOAT64:
                    res.float64_count += 1
                if t.dtype == DType.INT64:
                    res.int64_count += 1
                res.total_params += vol
                res.total_memory_bytes += vol * dtype_size(t.dtype)

    for n in graph.nodes:
        macs = 0
        flops = 0
        mem_bandwidth = 0

        # Ignored for compute ops
        if n.op_type in [
            "Gather",
            "Scatter",
            "ScatterND",
            "GatherND",
            "NonZero",
            "Shape",
            "Size",
            "Constant",
        ]:
            pass

        elif n.op_type in [
            "Reshape",
            "Transpose",
            "Squeeze",
            "Unsqueeze",
            "Expand",
            "Tile",
            "SpaceToDepth",
            "DepthToSpace",
            "Pad",
            "Resize",
            "Split",
            "Concat",
        ]:
            out_vol = (
                resolve_volume(graph.tensors[n.outputs[0]].shape, dynamic_overrides)
                if n.outputs and n.outputs[0] in graph.tensors
                else 0
            )
            in_vol = (
                resolve_volume(graph.tensors[n.inputs[0]].shape, dynamic_overrides)
                if n.inputs and n.inputs[0] in graph.tensors
                else 0
            )

            # Read + Write bytes
            if n.inputs and n.inputs[0] in graph.tensors:
                d_size = dtype_size(graph.tensors[n.inputs[0]].dtype)
                if isinstance(in_vol, int) and isinstance(out_vol, int):
                    mem_bandwidth = (in_vol + out_vol) * d_size
                else:
                    mem_bandwidth = f"({in_vol} + {out_vol}) * {d_size}"
        else:
            out_vol = 0
            if n.outputs and n.outputs[0] in graph.tensors:
                out_vol = resolve_volume(graph.tensors[n.outputs[0]].shape, dynamic_overrides)

            if n.op_type in ["Conv", "ConvTranspose"]:
                in_shape = graph.tensors[n.inputs[0]].shape if n.inputs else ()
                w_shape = graph.tensors[n.inputs[1]].shape if len(n.inputs) > 1 else ()
                out_shape = graph.tensors[n.outputs[0]].shape if n.outputs else ()

                out_v = resolve_volume(out_shape, dynamic_overrides)
                k_vol = resolve_volume(w_shape[2:], dynamic_overrides) if len(w_shape) > 2 else 1
                in_c = in_shape[1] if len(in_shape) > 1 else 1
                if isinstance(in_c, DynamicDim):
                    in_c = in_c.value
                if dynamic_overrides and in_c in dynamic_overrides:
                    in_c = dynamic_overrides[in_c]

                groups = 1
                for attr in n.attributes.values():
                    if attr.name == "group":
                        groups = attr.value
                        break

                if isinstance(out_v, int) and isinstance(k_vol, int) and isinstance(in_c, int):
                    macs = out_v * k_vol * (in_c // groups)
                    flops = macs * 2
                else:
                    macs = f"({out_v} * {k_vol} * {in_c} / {groups})"
                    flops = f"2 * {macs}"

            elif n.op_type in ["MatMul", "Gemm", "MatMulInteger", "QLinearMatMul", "QLinearConv"]:
                in1_shape = graph.tensors[n.inputs[0]].shape if n.inputs else ()
                k = in1_shape[-1] if len(in1_shape) > 0 else 1
                if isinstance(k, DynamicDim):
                    k = k.value
                if dynamic_overrides and k in dynamic_overrides:
                    k = dynamic_overrides[k]

                out_v = (
                    resolve_volume(graph.tensors[n.outputs[0]].shape, dynamic_overrides)
                    if n.outputs
                    else 1
                )

                if isinstance(out_v, int) and isinstance(k, int):
                    macs = out_v * k
                    flops = macs * 2
                else:
                    macs = f"({out_v} * {k})"
                    flops = f"2 * {macs}"

            elif n.op_type == "If":
                then_graph = get_attr(n, "then_branch")
                else_graph = get_attr(n, "else_branch")
                t_macs, t_flops, e_macs, e_flops = 0, 0, 0, 0
                if then_graph and isinstance(then_graph, Graph):
                    t_res = profile(then_graph, dynamic_overrides)
                    t_macs = t_res.total_macs
                    t_flops = t_res.total_flops
                if else_graph and isinstance(else_graph, Graph):
                    e_res = profile(else_graph, dynamic_overrides)
                    e_macs = e_res.total_macs
                    e_flops = e_res.total_flops

                # We record the worst-case (max) flops path
                if isinstance(t_flops, int) and isinstance(e_flops, int):
                    flops = max(t_flops, e_flops)
                    macs = max(t_macs, e_macs)
                else:
                    flops = f"max({t_flops}, {e_flops})"
                    macs = f"max({t_macs}, {e_macs})"

            elif n.op_type == "Loop":
                body_graph = get_attr(n, "body")
                m_tensor = graph.tensors.get(n.inputs[0]) if len(n.inputs) > 0 else None
                iters = 1
                if (
                    m_tensor
                    and hasattr(m_tensor, "values")
                    and m_tensor.values is not None
                    and len(m_tensor.values) > 0
                ):
                    iters = int(m_tensor.values[0])
                else:
                    iters = DynamicDim("loop_iters").value
                    if dynamic_overrides and iters in dynamic_overrides:
                        iters = dynamic_overrides[iters]

                b_macs, b_flops = 0, 0
                if body_graph and isinstance(body_graph, Graph):
                    b_res = profile(body_graph, dynamic_overrides)
                    b_macs = b_res.total_macs
                    b_flops = b_res.total_flops

                if isinstance(iters, int) and isinstance(b_flops, int):
                    flops = iters * b_flops
                    macs = iters * b_macs
                else:
                    flops = f"({iters} * {b_flops})"
                    macs = f"({iters} * {b_macs})"

            elif n.op_type in ["Attention", "MultiHeadAttention"]:
                # simplified transformer attention flops
                # FLOPs = 4 * batch * seq_len * embed_dim^2 + 2 * batch * seq_len^2 * embed_dim
                in1_shape = graph.tensors[n.inputs[0]].shape if n.inputs else ()
                if len(in1_shape) >= 3:
                    b = resolve_volume((in1_shape[0],), dynamic_overrides)
                    s = resolve_volume((in1_shape[1],), dynamic_overrides)
                    e = resolve_volume((in1_shape[2],), dynamic_overrides)
                    if isinstance(b, int) and isinstance(s, int) and isinstance(e, int):
                        macs = 2 * b * s * e * e + b * s * s * e
                        flops = macs * 2
                    else:
                        macs = f"(2 * {b} * {s} * {e}^2 + {b} * {s}^2 * {e})"
                        flops = f"2 * {macs}"

            elif n.op_type in ["RNN", "LSTM", "GRU"]:
                seq_len = (
                    resolve_volume(graph.tensors[n.inputs[0]].shape[:1], dynamic_overrides)
                    if n.inputs and len(graph.tensors[n.inputs[0]].shape) > 0
                    else 1
                )
                batch = (
                    resolve_volume(graph.tensors[n.inputs[0]].shape[1:2], dynamic_overrides)
                    if n.inputs and len(graph.tensors[n.inputs[0]].shape) > 1
                    else 1
                )
                hidden = (
                    resolve_volume(graph.tensors[n.inputs[2]].shape[2:3], dynamic_overrides)
                    if len(n.inputs) > 2 and len(graph.tensors[n.inputs[2]].shape) > 2
                    else 1
                )
                input_size = (
                    resolve_volume(graph.tensors[n.inputs[0]].shape[2:3], dynamic_overrides)
                    if n.inputs and len(graph.tensors[n.inputs[0]].shape) > 2
                    else 1
                )

                gates = 1
                if n.op_type == "LSTM":
                    gates = 4
                elif n.op_type == "GRU":
                    gates = 3

                if (
                    isinstance(seq_len, int)
                    and isinstance(batch, int)
                    and isinstance(hidden, int)
                    and isinstance(input_size, int)
                ):
                    macs = seq_len * batch * gates * hidden * (input_size + hidden)
                    flops = macs * 2
                else:
                    macs = f"({seq_len} * {batch} * {gates} * {hidden} * ({input_size} + {hidden}))"
                    flops = f"2 * {macs}"

            elif n.op_type in [
                "Add",
                "Sub",
                "Mul",
                "Div",
                "Relu",
                "Sigmoid",
                "Tanh",
                "Gelu",
                "Exp",
                "Log",
                "Sin",
                "Cos",
                "Round",
                "IsInf",
                "IsNaN",
                "BitwiseAnd",
                "BitwiseOr",
                "BitwiseXor",
                "BitwiseNot",
                "Hardmax",
                "LogSoftmax",
                "HardSigmoid",
                "HardSwish",
                "Shrink",
                "PRelu",
            ]:
                flops = out_vol

            elif n.op_type in ["BatchNormalization", "LayerNormalization", "InstanceNormalization"]:
                if isinstance(out_vol, int):
                    flops = out_vol * 4
                else:
                    flops = f"4 * {out_vol}"

            elif n.op_type in ["ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "Softmax"]:
                in_vol = (
                    resolve_volume(graph.tensors[n.inputs[0]].shape, dynamic_overrides)
                    if n.inputs
                    else out_vol
                )
                flops = in_vol

            # Memory bandwidth logic for math ops (reads inputs, writes output)
            if n.inputs and n.outputs:
                read_bytes = sum(
                    [
                        resolve_volume(graph.tensors[inp].shape, dynamic_overrides)
                        * dtype_size(graph.tensors[inp].dtype)
                        for inp in n.inputs
                        if inp in graph.tensors
                        and isinstance(
                            resolve_volume(graph.tensors[inp].shape, dynamic_overrides), int
                        )
                    ]
                )
                write_bytes = 0
                if n.outputs[0] in graph.tensors:
                    v = resolve_volume(graph.tensors[n.outputs[0]].shape, dynamic_overrides)
                    if isinstance(v, int):
                        write_bytes = v * dtype_size(graph.tensors[n.outputs[0]].dtype)
                if isinstance(read_bytes, int) and isinstance(write_bytes, int):
                    mem_bandwidth = read_bytes + write_bytes

        # Precision tracking
        dt = DType.FLOAT32
        if n.inputs and n.inputs[0] in graph.tensors:
            dt = graph.tensors[n.inputs[0]].dtype

        if dt == DType.FLOAT32:
            res.fp32_macs = _add_metric(res.fp32_macs, macs)
        elif dt in [DType.FLOAT16, DType.BFLOAT16]:
            res.fp16_macs = _add_metric(res.fp16_macs, macs)
        elif dt == DType.INT8:
            res.int8_macs = _add_metric(res.int8_macs, macs)

        intensity = "N/A"
        if isinstance(flops, int) and isinstance(mem_bandwidth, int) and mem_bandwidth > 0:
            intensity = round(flops / mem_bandwidth, 2)

        node_params = 0
        node_activations = 0
        for inp in n.inputs:
            if inp in graph.tensors and (
                isinstance(graph.tensors[inp], Constant) or graph.tensors[inp].is_initializer
            ):
                v = resolve_volume(graph.tensors[inp].shape, dynamic_overrides)
                if isinstance(v, int):
                    node_params += v * dtype_size(graph.tensors[inp].dtype)
        for out in n.outputs:
            if out in graph.tensors:
                v = resolve_volume(graph.tensors[out].shape, dynamic_overrides)
                if isinstance(v, int):
                    node_activations += v * dtype_size(graph.tensors[out].dtype)

        res.node_profiles.append(
            {
                "name": n.name,
                "op_type": n.op_type,
                "macs": macs,
                "flops": flops,
                "mem_bandwidth_bytes": mem_bandwidth,
                "arithmetic_intensity": intensity,
                "params": node_params,
                "activation_bytes": node_activations,
            }
        )
        res.total_macs = _add_metric(res.total_macs, macs)
        res.total_flops = _add_metric(res.total_flops, flops)

    activation_bytes = 0
    for n in graph.nodes:
        for out_name in n.outputs:
            if out_name in graph.tensors:
                t = graph.tensors[out_name]
                vol = resolve_volume(t.shape, dynamic_overrides)
                if isinstance(vol, int):
                    activation_bytes += vol * dtype_size(t.dtype)
    res.peak_activation_bytes = activation_bytes

    return res


Graph.profile = profile
