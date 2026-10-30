/**
 * @fileoverview clip.ts
 * Provides clip functionality for the core package.
 */
import { Tensor } from '../ir/tensor.js';
import { recordOp } from '../macros.js';
import { Gemm } from '../primitives.js';
import { LLaMA } from './llama.js';
import { VisionTransformer } from './vit.js';

/**
 * Executes `getParam`.
 * @param name Parameter `name`.
 * @param shape Parameter `shape`.
 * @param dtype Parameter `dtype`.
 * @returns Result of `getParam`.
 */
function getParam(
  name: string,
  shape: number[],
  dtype: ReturnType<typeof JSON.parse> = 'float32',
): Tensor {
  return new Tensor(name, shape, dtype, false, false, new Float32Array());
}

/** Represents the `CLIP` class. */
export class CLIP {
  /** Property `embedDim` of `CLIP`. */
  public embedDim: number;
  /** Property `visual` of `CLIP`. */
  public visual: VisionTransformer;
  /** Property `text` of `CLIP`. */
  public text: LLaMA;
  /** Property `textProjection` of `CLIP`. */
  public textProjection: Gemm;
  /** Property `logitScale` of `CLIP`. */
  public logitScale: Tensor;

  /**
   * Initializes a new `CLIP` instance.
   * @param embedDim Parameter `embedDim`.
   * @param visionWidth Parameter `visionWidth`.
   * @param visionLayers Parameter `visionLayers`.
   * @param visionHeads Parameter `visionHeads`.
   * @param textWidth Parameter `textWidth`.
   * @param textLayers Parameter `textLayers`.
   * @param textHeads Parameter `textHeads`.
   * @param vocabSize Parameter `vocabSize`.
   */
  constructor(
    embedDim: number = 512,
    visionWidth: number = 768,
    visionLayers: number = 12,
    visionHeads: number = 12,
    textWidth: number = 512,
    textLayers: number = 12,
    textHeads: number = 8,
    vocabSize: number = 49408,
  ) {
    this.embedDim = embedDim;
    this.visual = new VisionTransformer(
      224,
      16,
      3,
      embedDim,
      visionWidth,
      visionLayers,
      visionHeads,
    );
    this.text = new LLaMA(vocabSize, textWidth, textHeads, textHeads, textLayers);
    this.textProjection = new Gemm(1.0, 1.0, 0, 1);
    this.logitScale = getParam('logit_scale', [1]);
  }

  /**
   * Executes `call` on `CLIP`.
   * @param image Parameter `image`.
   * @param text Parameter `text`.
   * @returns Result of `call`.
   */
  call(image: Tensor, text: Tensor): [Tensor, Tensor] {
    const imageFeatures = this.visual.call(image);

    const pos = recordOp('Constant', [], { value: [0], dtype: 7 });
    let textFeatures = this.text.call(text, pos);

    textFeatures = this.textProjection.call(
      textFeatures,
      getParam('text_projection.weight', [this.embedDim, this.text.dim]),
    );

    return [imageFeatures, textFeatures];
  }
}

/**
 * Executes `clipVitBasePatch16`.
 * @returns Result of `clipVitBasePatch16`.
 */
export function clipVitBasePatch16(): CLIP {
  return new CLIP();
}
