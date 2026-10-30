"""WebGPU Shading Language (WGSL) compute shader generator for ONNX9000.

Translates ONNX9000 IR nodes and graphs into WebGPU compute shaders and calculates
optimal 3D workgroup dispatch dimensions.
"""

from onnx9000.core.ir import DynamicDim, Graph, Node


class WGSLGenerator:
    """Generates WebGPU Shading Language (WGSL) compute kernels from ONNX9000 IR nodes."""

    def generate_shader(self, node: Node) -> str:
        """Map an ONNX9000 Node to a WGSL compute shader string.

        Args:
            node: The ONNX9000 IR Node to compile into WGSL.

        Returns:
            Complete WGSL compute shader source string.
        """
        op = node.op_type.lower()
        shader_lines = [
            "@group(0) @binding(0) var<storage, read> input_a : array<f32>;",
            "@group(0) @binding(1) var<storage, read_write> output : array<f32>;",
            "@compute @workgroup_size(64)",
            "fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {",
            "    let idx = global_id.x;",
        ]
        if op == "relu":
            shader_lines.append("    output[idx] = max(0.0, input_a[idx]);")
        elif op == "abs":
            shader_lines.append("    output[idx] = abs(input_a[idx]);")
        else:
            shader_lines.append(f"    // op: {op}")
            shader_lines.append("    output[idx] = input_a[idx];")
        shader_lines.append("}")
        return chr(10).join(shader_lines)

    def generate_graph_shaders(self, graph: Graph) -> dict[str, str]:
        """Generate WGSL compute shaders for every node in an ONNX9000 Graph.

        Args:
            graph: The ONNX9000 Graph to compile.

        Returns:
            Dictionary mapping each node name to its WGSL shader source.
        """
        shaders: dict[str, str] = {}
        for idx, node in enumerate(graph.nodes):
            key = node.name or f"{node.op_type}_{idx}"
            shaders[key] = self.generate_shader(node)
        return shaders

    def compute_workgroup_size(
        self, shape: tuple[int | DynamicDim, ...] | list[int]
    ) -> tuple[int, int, int]:
        """Calculate 3D WebGPU workgroup sizes bounded by 256 per axis based on tensor shape.

        Args:
            shape: Tensor shape dimensions.

        Returns:
            3D workgroup size tuple `(x, y, z)` where each component is in `[1, 256]`.
        """
        if not shape:
            return (1, 1, 1)
        int_dims = [d if isinstance(d, int) and d > 0 else 1 for d in shape]
        x = int_dims[-1]
        y = int_dims[-2] if len(int_dims) >= 2 else 1
        z = int_dims[-3] if len(int_dims) >= 3 else 1
        return (min(x, 256), min(y, 256), min(z, 256))


__all__ = ["WGSLGenerator"]
