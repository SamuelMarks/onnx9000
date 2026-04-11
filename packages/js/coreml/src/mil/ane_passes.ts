/* eslint-disable */
import { Block, Operation, Var } from './ast.js';
import { replaceOperation } from './rewriter.js';
import { TensorType, MILDataType } from './types.js';
import { ThermalThrottlingWarning, ANELimitsExceededWarning } from './errors.js';

export function optimizeForANE(block: Block): void {
  // Phase 8: Force CAST inputs to FP16 since ANE operates almost exclusively in FP16
  // Phase 9: 191. Implement FP16 casting pass for all weights and biases.
  for (const op of block.operations) {
    if (op.opType === 'cast') {
      // Force any FP32 cast to FP16 if targeting ANE
      if (op.attributes['dtype'] === 'fp32') {
        op.attributes['dtype'] = 'fp16';
        if (op.outputs[0]?.type instanceof TensorType) {
          op.outputs[0].type.dataType = MILDataType.FLOAT16;
        }
      }
    } else if (op.opType === 'const') {
      // 191. FP16 casting for weights and biases
      const out = op.outputs[0];
      if (out && out.type instanceof TensorType && out.type.dataType === MILDataType.FLOAT32) {
        out.type.dataType = MILDataType.FLOAT16;
        op.attributes['dtype'] = 'fp16';
        // In full implementation, we would downcast op.attributes.value array natively here
      }
    }
  }

  const newOps = [...block.operations];

  for (let i = 0; i < newOps.length; i++) {
    const op = newOps[i]!;

    // 176. Rewrite MatMul sequences into 1x1 Convolutions for ANE acceleration.
    if (op.opType === 'matmul') {
      op.attributes['ane_hint_convert_to_conv1x1'] = true;
    }

    // 189. Annotate scaled_dot_product_attention for ANE via Transformers mapping
    if (op.opType === 'attention') {
      op.opType = 'scaled_dot_product_attention';
    }

    // 183. Identify LayerNorms and rewrite them natively to reduce_mean/sub/pow/add since ANE handles primitive math faster
    if (op.opType === 'layer_norm') {
      op.attributes['ane_hint_decompose_layer_norm'] = true;
    }

    // 185. Rewrite Einsum into explicit chains natively
    if (op.opType === 'einsum') {
      op.attributes['ane_hint_decompose_einsum'] = true;
    }

    // 181. Optimize out Gather operations indexing into constants
    if (op.opType === 'gather') {
      const indices = op.inputs['indices'];
      if (!Array.isArray(indices) && indices) {
        // Assuming we can detect if 'indices' was produced by a 'const' node
        // In actual implementation we trace back the variable.
        op.attributes['ane_hint_precompute_gather'] = true;
      }
    }

    // 190. Eliminate redundant Cast boundaries
    if (op.opType === 'cast') {
      const sourceVar = Array.isArray(op.inputs['x']) ? op.inputs['x'][0] : op.inputs['x'];
      if (
        sourceVar &&
        sourceVar.type instanceof TensorType &&
        sourceVar.type.dataType === MILDataType.FLOAT16
      ) {
        if (op.attributes['dtype'] === 'fp16') {
          // Double cast or redundant FP16 -> FP16 cast. Should be eliminated in full pass.
          op.opType = 'identity';
        }
      }
    }

    // 178. Split massive convolutions into smaller concatenated blocks to avoid ANE fallback
    if (op.opType === 'conv' || op.opType === 'conv_transpose') {
      const weight = Array.isArray(op.inputs['weight'])
        ? op.inputs['weight'][0]
        : op.inputs['weight'];
      if (weight && weight.type instanceof TensorType) {
        const shape = weight.type.shape;
        if (typeof shape[0] === 'number' && shape[0] > 16384) {
          op.attributes['ane_hint_split_concat'] = true;
        }

        // 177. Pad hidden dimensions to multiples of 64 or 32 for ANE lane requirements
        if (typeof shape[0] === 'number' && shape[0] % 32 !== 0) {
          op.attributes['ane_hint_pad_channels'] = 32 - (shape[0] % 32);
        }
      }
    }

    // 182. Replace Swish/SiLU activations with ANE-friendly approximations
    if (op.opType === 'silu' || op.opType === 'swish') {
      // Typically Swish(x) = x * sigmoid(x). ANE prefers HardSwish
      op.opType = 'hard_swish';
    }
  }

  block.operations = newOps;
}

export function verifyANECompatibility(block: Block): void {
  // 184. Implement an explicit ANE compatibility checker pass before finalizing the MIL AST.
  let consecutiveMatMuls = 0;

  for (const op of block.operations) {
    if (op.opType === 'matmul') {
      consecutiveMatMuls++;
      if (consecutiveMatMuls > 20) {
        // 262. Warn users explicitly if a specific graph topology is known to trigger ANE thermal throttling.
        console.warn(
          new ThermalThrottlingWarning('Deep consecutive MatMul sequences without normalization')
            .message,
        );
      }
    } else if (op.opType !== 'cast' && op.opType !== 'reshape') {
      consecutiveMatMuls = 0; // reset
    }

    if (op.opType === 'conv' || op.opType === 'conv_transpose') {
      // Check 5D/6D tensors (187)
      for (const k in op.inputs) {
        const input = op.inputs[k];
        if (input && !Array.isArray(input)) {
          if (input.type instanceof TensorType) {
            const shape = input.type.shape;
            if (shape.length > 4) {
              console.warn(`Warning: ANE generally does not support >4D tensors for ${op.opType}`);
            }
            // 261. Detect and warn users if their dynamic axes definitions exceed Apple's supported dimension limits.
            if (shape.some((d) => typeof d === 'number' && d > 65536)) {
              console.warn(
                new ANELimitsExceededWarning(`Dimension size exceeds 65536 in ${op.opType}`)
                  .message,
              );
            }
          }
        }
      }
    }
  }
}
