/**
 * @fileoverview Model architecture registry and block builders for `@onnx9000/new-model-arch`.
 * Supports modern architectures including LLaMA, Mamba, RWKV, Mixtral, ViT, Swin, ConvNeXt, DiT, CLIP, and Whisper.
 */

/**
 * Supported modern neural architecture families.
 */
export type ModelArchitectureFamily =
  | 'llama'
  | 'mamba'
  | 'rwkv'
  | 'mixtral'
  | 'vit'
  | 'swin'
  | 'convnext'
  | 'dit'
  | 'clip'
  | 'whisper';

/**
 * Architectural specification for building a model backbone.
 */
export interface ArchitectureSpec {
  /** Architecture family identifier. */
  family: ModelArchitectureFamily;
  /** Hidden embedding dimension. */
  hiddenSize: number;
  /** Number of repeated backbone blocks. */
  numBlocks: number;
  /** Sequence of primary ONNX operator types composing each block. */
  blockOps: string[];
}

/**
 * Registry and builder for modern neural network architectures.
 */
export class NewModelArch {
  /** Canonical block operator templates per architecture family. */
  private static readonly FAMILY_OPS: Readonly<Record<ModelArchitectureFamily, string[]>> = {
    llama: ['RMSNormalization', 'RotaryEmbedding', 'GroupQueryAttention', 'SwiGLU'],
    mamba: ['RMSNormalization', 'Conv1D', 'SelectiveScanSSM', 'Linear'],
    rwkv: ['LayerNormalization', 'TimeMixWKV', 'ChannelMixReLU2'],
    mixtral: ['RMSNormalization', 'GroupQueryAttention', 'Top2SparseMoE'],
    vit: ['ConvPatchEmbed', 'LayerNormalization', 'MultiHeadAttention', 'MLP'],
    swin: ['PatchPartition', 'WindowMultiHeadAttention', 'ShiftedWindowAttention', 'PatchMerging'],
    convnext: ['DepthwiseConv7x7', 'LayerNormalization', 'PointwiseConv1x1', 'Gelu'],
    dit: ['AdaLayerNormZero', 'MultiHeadAttention', 'PointwiseFeedForward'],
    clip: ['VisionTransformer', 'TextTransformer', 'L2Normalize', 'CosineSimilarity'],
    whisper: ['Conv1DStem', 'EncoderSelfAttention', 'DecoderCrossAttention'],
  };

  /**
   * Builds an `ArchitectureSpec` for a given model family and size.
   * @param family Model architecture family.
   * @param hiddenSize Hidden dimension (defaults to 768).
   * @param numBlocks Number of repeated blocks (defaults to 12).
   * @returns Populated `ArchitectureSpec`.
   */
  public buildSpec(
    family: ModelArchitectureFamily,
    hiddenSize = 768,
    numBlocks = 12,
  ): ArchitectureSpec {
    return {
      family,
      hiddenSize,
      numBlocks,
      blockOps: [...NewModelArch.FAMILY_OPS[family]],
    };
  }

  /**
   * Processes a model architecture request string.
   * @param arg Architecture identifier string.
   * @returns Formatted processing status string.
   */
  public process(arg: string): string {
    this.buildSpec('mamba');
    return `New Model Arch processed ${arg}`;
  }
}
