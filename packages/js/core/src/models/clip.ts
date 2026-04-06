import { Tensor } from '../ir/tensor.js';
import { Gemm } from '../primitives.js';
import { VisionTransformer } from './vit.js';
import { LLaMA } from './llama.js';

function getParam(
  name: string,
  shape: number[],
  dtype: ReturnType<typeof JSON.parse> = 'float32',
): Tensor {
  return new Tensor(name, shape, dtype, false, false, new Float32Array());
}

function recordOp(opType: string, inputs: Tensor[], attr?: ReturnType<typeof JSON.parse>): Tensor {
  const dtype = inputs[0]?.dtype ?? 'float32';
  return new Tensor(`${opType}_out`, [], dtype, false, false, new Float32Array());
}

export class CLIP {
  public embedDim: number;
  public visual: VisionTransformer;
  public text: LLaMA;
  public textProjection: Gemm;
  public logitScale: Tensor;

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

export function clipVitBasePatch16(): CLIP {
  return new CLIP();
}
