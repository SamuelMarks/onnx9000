from onnx9000.core.ir import Attribute, DType, Graph, Node, Tensor


class GGUFQuantizationMapper:
    """Map GGUF block-quantization scales strictly to Core IR DequantizeLinear with block-size attributes."""

    def map_block(self, graph: Graph, gguf_qtype: str, weight_name: str, block_size: int) -> None:
        """Map GGUF types like Q4_K_M, Q8_0."""
        # Insert DequantizeLinear
        dequant = Node(
            op_type="DequantizeLinear",
            inputs=[weight_name, f"{weight_name}_scale", f"{weight_name}_zero_point"],
            outputs=[f"{weight_name}_fp32"],
        )
        dequant.attributes["block_size"] = Attribute("block_size", "int", block_size)
        dequant.attributes["gguf_type"] = Attribute("gguf_type", "string", gguf_qtype)
        graph.nodes.append(dequant)


class AWQParser:
    """Parse Hugging Face AWQ config files to reconstruct group-wise INT4 weights."""

    def parse_config(self, graph: Graph, config: dict, weight_name: str) -> None:
        group_size = config.get("group_size", 128)
        # Register group-wise quantization
        dequant = Node(
            op_type="DequantizeLinear",
            inputs=[weight_name, f"{weight_name}_awq_scales", f"{weight_name}_awq_zeros"],
            outputs=[f"{weight_name}_fp16"],
        )
        dequant.attributes["block_size"] = Attribute("block_size", "int", group_size)
        graph.nodes.append(dequant)


class GPTQParser:
    """Parse GPTQ state-dicts and descramble act-order permutations in the IR."""

    def parse_state_dict(self, graph: Graph, state_dict: dict, weight_name: str) -> None:
        g_idx = state_dict.get(f"{weight_name}.g_idx")
        if g_idx is not None:
            # Descramble pass logic using Gather
            gather = Node(
                op_type="Gather",
                inputs=[f"{weight_name}_quant", f"{weight_name}.g_idx"],
                outputs=[f"{weight_name}_descrambled"],
            )
            gather.attributes["axis"] = Attribute("axis", "int", 0)
            graph.nodes.append(gather)


class AQTParser:
    """Replicate Bonsai AQT INT8/INT4 symmetric bound tensors."""

    def parse_aqt(self, graph: Graph, weight_name: str, bitwidth: int = 8) -> None:
        dequant = Node(
            op_type="DequantizeLinear",
            inputs=[weight_name, f"{weight_name}_scale"],
            outputs=[f"{weight_name}_fp32"],
        )
        # Symmetric lacks zero_point
        dequant.attributes["symmetric"] = Attribute("symmetric", "int", 1)
        dequant.attributes["bitwidth"] = Attribute("bitwidth", "int", bitwidth)
        graph.nodes.append(dequant)
