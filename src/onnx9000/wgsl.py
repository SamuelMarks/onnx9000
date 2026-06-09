"""WGSL Generator."""

from ml_switcheroo_ir import LogicalGraph, LogicalNode


class WGSLGenerator:
    def generate_shader(self, node: LogicalNode) -> str:
        """Map ONNX nodes to WebGPU Shading Language (WGSL) strings."""
        if node.op_type == "Add":
            return "fn op(a: f32, b: f32) -> f32 { return a + b; }"
        elif node.op_type == "Mul":
            return "fn op(a: f32, b: f32) -> f32 { return a * b; }"
        return "fn op(a: f32, b: f32) -> f32 { return 0.0; }"

    def generate_bindings(self) -> str:
        """Generate standard BindGroup and PipelineLayout JavaScript glue code."""
        return """
const bindGroupLayout = device.createBindGroupLayout({
    entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
    ]
});
"""

    def calculate_workgroup_size(self, shape) -> tuple:
        """Calculate workgroup_size dynamically based on inferred tensor shapes."""
        if not shape:
            return (1, 1, 1)
        x = shape[-1] if len(shape) >= 1 else 1
        y = shape[-2] if len(shape) >= 2 else 1
        z = shape[-3] if len(shape) >= 3 else 1
        # Simple mock scaling down to max workgroup size (e.g. 256)
        return (min(x, 256), min(y, 256), min(z, 256))
