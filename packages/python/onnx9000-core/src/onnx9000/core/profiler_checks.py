"""Provides profiler checks module functionality."""

from onnx9000.core.ir import Constant, Graph


class OptimizationAnalyzer:
    """Represents the Optimization Analyzer class."""

    def __init__(self, graph: Graph):
        """Initializes the instance."""
        self.graph = graph
        self.opportunities = []

    def analyze(self):
        """Executes the analyze operation."""
        self._check_redundant_casts()
        self._check_unused_initializers()
        self._check_fusion_matmul_add()
        self._check_fusion_conv_bn()
        self._check_identity_chains()
        self._check_unsupported_webgpu()
        return self.opportunities

    def _check_redundant_casts(self):
        """Executes the check redundant casts operation."""
        for n in self.graph.nodes:
            if n.op_type == "Cast":
                # if input dtype == output dtype
                pass  # simplification for now

    def _check_unused_initializers(self):
        """Executes the check unused initializers operation."""
        used_inputs = set()
        for n in self.graph.nodes:
            for inp in n.inputs:
                used_inputs.add(inp)

        unused = []
        for t_name, t in self.graph.tensors.items():
            if (
                (isinstance(t, Constant) or t.is_initializer)
                and t_name not in used_inputs
                and t_name not in self.graph.outputs
            ):
                unused.append(t_name)

        if unused:
            self.opportunities.append(
                f"Found {len(unused)} completely unused initializers/constants. They can be safely removed."
            )

    def _check_fusion_matmul_add(self):
        # MatMul -> Add
        """Executes the check fusion matmul add operation."""
        count = 0
        for n in self.graph.nodes:
            if n.op_type == "MatMul":
                if n.outputs and n.outputs[0] in self.graph.tensors:
                    consumers = [c for c in self.graph.nodes if n.outputs[0] in c.inputs]
                    if len(consumers) == 1 and consumers[0].op_type == "Add":
                        count += 1
        if count > 0:
            self.opportunities.append(
                f"Found {count} un-fused MatMul + Add structures. Consider fusing them into Gemm."
            )

    def _check_fusion_conv_bn(self):
        """Executes the check fusion conv bn operation."""
        count = 0
        for n in self.graph.nodes:
            if n.op_type == "Conv":
                if n.outputs and n.outputs[0] in self.graph.tensors:
                    consumers = [c for c in self.graph.nodes if n.outputs[0] in c.inputs]
                    if len(consumers) == 1 and consumers[0].op_type == "BatchNormalization":
                        count += 1
        if count > 0:
            self.opportunities.append(
                f"Found {count} missing Conv + BatchNorm fusion opportunities."
            )

    def _check_identity_chains(self):
        """Executes the check identity chains operation."""
        count = 0
        for n in self.graph.nodes:
            if n.op_type == "Identity":
                count += 1
        if count > 0:
            self.opportunities.append(
                f"Found {count} Identity operations. These can usually be eliminated."
            )

    def _check_unsupported_webgpu(self):
        """Executes the check unsupported webgpu operation."""
        unsupported = ["Loop", "Scan", "SequenceConstruct", "Complex", "NonZero"]
        found = set()
        for n in self.graph.nodes:
            if n.op_type in unsupported:
                found.add(n.op_type)
        for f in found:
            self.opportunities.append(
                f"Flag: Operator '{f}' is generally unsupported or very slow on WebGPU compute shaders."
            )
