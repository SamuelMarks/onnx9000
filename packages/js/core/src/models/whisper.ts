import { Tensor } from '../ir/tensor.js';
import { ConvND, Gemm, LayerNormalization, MultiHeadAttention, Gelu } from '../primitives.js';

function getParam(name: string, shape: number[], dtype: any = 'float32'): Tensor {
  return new Tensor(name, shape, dtype, false, false, new Float32Array());
}

function recordOp(opType: string, inputs: Tensor[], attr?: any): Tensor {
  const dtype = inputs[0]?.dtype ?? 'float32';
  return new Tensor(`${opType}_out`, [], dtype, false, false, new Float32Array());
}

export class WhisperEncoderLayer {
  public prefix: string;
  public dModel: number;
  public selfAttn: MultiHeadAttention;
  public selfAttnLayerNorm: LayerNormalization;
  public fc1: Gemm;
  public act: Gelu;
  public fc2: Gemm;
  public finalLayerNorm: LayerNormalization;

  constructor(
    dModel: number,
    encoderAttentionHeads: number,
    encoderFfnDim: number,
    prefix: string = '',
  ) {
    this.prefix = prefix;
    this.dModel = dModel;
    this.selfAttn = new MultiHeadAttention(encoderAttentionHeads);
    this.selfAttnLayerNorm = new LayerNormalization([dModel]);
    this.fc1 = new Gemm(1.0, 1.0, 0, 1);
    this.act = new Gelu();
    this.fc2 = new Gemm(1.0, 1.0, 0, 1);
    this.finalLayerNorm = new LayerNormalization([dModel]);
  }

  call(x: Tensor): Tensor {
    let identity = x;
    let xNorm = this.selfAttnLayerNorm.call(
      x,
      getParam(`${this.prefix}.self_attn_layer_norm.weight`, [this.dModel]),
      getParam(`${this.prefix}.self_attn_layer_norm.bias`, [this.dModel]),
    );
    let xAttn = this.selfAttn.call(xNorm, xNorm, xNorm);
    x = recordOp('Add', [identity, xAttn]);

    identity = x;
    xNorm = this.finalLayerNorm.call(
      x,
      getParam(`${this.prefix}.final_layer_norm.weight`, [this.dModel]),
      getParam(`${this.prefix}.final_layer_norm.bias`, [this.dModel]),
    );
    let xFfn = this.fc1.call(
      xNorm,
      getParam(`${this.prefix}.fc1.weight`, [this.dModel * 4, this.dModel]),
      getParam(`${this.prefix}.fc1.bias`, [this.dModel * 4]),
    );
    xFfn = this.act.call(xFfn);
    xFfn = this.fc2.call(
      xFfn,
      getParam(`${this.prefix}.fc2.weight`, [this.dModel, this.dModel * 4]),
      getParam(`${this.prefix}.fc2.bias`, [this.dModel]),
    );
    return recordOp('Add', [identity, xFfn]);
  }
}

export class WhisperEncoder {
  public dModel: number;
  public conv1: ConvND;
  public act1: Gelu;
  public conv2: ConvND;
  public act2: Gelu;
  public layers: WhisperEncoderLayer[];
  public layerNorm: LayerNormalization;

  constructor(
    dModel: number = 512,
    encoderAttentionHeads: number = 8,
    encoderFfnDim: number = 2048,
    encoderLayers: number = 6,
  ) {
    this.dModel = dModel;
    this.conv1 = new ConvND(1, 80, dModel, 3, 1, 1, 1, 1, false);
    this.act1 = new Gelu();
    this.conv2 = new ConvND(1, dModel, dModel, 3, 2, 1, 1, 1, false);
    this.act2 = new Gelu();

    this.layers = [];
    for (let i = 0; i < encoderLayers; i++) {
      this.layers.push(
        new WhisperEncoderLayer(dModel, encoderAttentionHeads, encoderFfnDim, `layers.${i}`),
      );
    }
    this.layerNorm = new LayerNormalization([dModel]);
  }

  call(x: Tensor): Tensor {
    x = this.conv1.call(
      x,
      getParam('conv1.weight', [this.dModel, 80, 3]),
      getParam('conv1.bias', [this.dModel]),
    );
    x = this.act1.call(x);
    x = this.conv2.call(
      x,
      getParam('conv2.weight', [this.dModel, this.dModel, 3]),
      getParam('conv2.bias', [this.dModel]),
    );
    x = this.act2.call(x);

    x = recordOp('Transpose', [x], { perm: [0, 2, 1] });

    const posEmbed = getParam('embed_positions.weight', [1, 1500, this.dModel]);
    x = recordOp('Add', [x, posEmbed]);

    for (const layer of this.layers) {
      x = layer.call(x);
    }

    x = this.layerNorm.call(
      x,
      getParam('layer_norm.weight', [this.dModel]),
      getParam('layer_norm.bias', [this.dModel]),
    );
    return x;
  }
}

export class WhisperDecoderLayer {
  public prefix: string;
  public dModel: number;
  public selfAttn: MultiHeadAttention;
  public selfAttnLayerNorm: LayerNormalization;
  public encoderAttn: MultiHeadAttention;
  public encoderAttnLayerNorm: LayerNormalization;
  public fc1: Gemm;
  public act: Gelu;
  public fc2: Gemm;
  public finalLayerNorm: LayerNormalization;

  constructor(
    dModel: number,
    decoderAttentionHeads: number,
    decoderFfnDim: number,
    prefix: string = '',
  ) {
    this.prefix = prefix;
    this.dModel = dModel;
    this.selfAttn = new MultiHeadAttention(decoderAttentionHeads);
    this.selfAttnLayerNorm = new LayerNormalization([dModel]);
    this.encoderAttn = new MultiHeadAttention(decoderAttentionHeads);
    this.encoderAttnLayerNorm = new LayerNormalization([dModel]);
    this.fc1 = new Gemm(1.0, 1.0, 0, 1);
    this.act = new Gelu();
    this.fc2 = new Gemm(1.0, 1.0, 0, 1);
    this.finalLayerNorm = new LayerNormalization([dModel]);
  }

  call(x: Tensor, encoderHiddenStates: Tensor, causalMask?: Tensor): Tensor {
    let identity = x;
    let xNorm = this.selfAttnLayerNorm.call(
      x,
      getParam(`${this.prefix}.self_attn_layer_norm.weight`, [this.dModel]),
      getParam(`${this.prefix}.self_attn_layer_norm.bias`, [this.dModel]),
    );
    let xAttn = this.selfAttn.call(xNorm, xNorm, xNorm, causalMask);
    x = recordOp('Add', [identity, xAttn]);

    identity = x;
    xNorm = this.encoderAttnLayerNorm.call(
      x,
      getParam(`${this.prefix}.encoder_attn_layer_norm.weight`, [this.dModel]),
      getParam(`${this.prefix}.encoder_attn_layer_norm.bias`, [this.dModel]),
    );
    xAttn = this.encoderAttn.call(xNorm, encoderHiddenStates, encoderHiddenStates);
    x = recordOp('Add', [identity, xAttn]);

    identity = x;
    xNorm = this.finalLayerNorm.call(
      x,
      getParam(`${this.prefix}.final_layer_norm.weight`, [this.dModel]),
      getParam(`${this.prefix}.final_layer_norm.bias`, [this.dModel]),
    );
    let xFfn = this.fc1.call(
      xNorm,
      getParam(`${this.prefix}.fc1.weight`, [this.dModel * 4, this.dModel]),
      getParam(`${this.prefix}.fc1.bias`, [this.dModel * 4]),
    );
    xFfn = this.act.call(xFfn);
    xFfn = this.fc2.call(
      xFfn,
      getParam(`${this.prefix}.fc2.weight`, [this.dModel, this.dModel * 4]),
      getParam(`${this.prefix}.fc2.bias`, [this.dModel]),
    );
    return recordOp('Add', [identity, xFfn]);
  }
}

export class WhisperDecoder {
  public vocabSize: number;
  public dModel: number;
  public layers: WhisperDecoderLayer[];
  public layerNorm: LayerNormalization;
  public lmHead: Gemm;

  constructor(
    vocabSize: number = 51865,
    dModel: number = 512,
    decoderAttentionHeads: number = 8,
    decoderFfnDim: number = 2048,
    decoderLayers: number = 6,
  ) {
    this.vocabSize = vocabSize;
    this.dModel = dModel;

    this.layers = [];
    for (let i = 0; i < decoderLayers; i++) {
      this.layers.push(
        new WhisperDecoderLayer(dModel, decoderAttentionHeads, decoderFfnDim, `layers.${i}`),
      );
    }
    this.layerNorm = new LayerNormalization([dModel]);
    this.lmHead = new Gemm(1.0, 1.0, 0, 1);
  }

  call(inputIds: Tensor, encoderHiddenStates: Tensor, causalMask?: Tensor): Tensor {
    let x = recordOp(
      'Gather',
      [getParam('embed_tokens.weight', [this.vocabSize, this.dModel]), inputIds],
      { axis: 0 },
    );
    const posEmbed = getParam('embed_positions.weight', [1, 448, this.dModel]);
    x = recordOp('Add', [x, posEmbed]);

    for (const layer of this.layers) {
      x = layer.call(x, encoderHiddenStates, causalMask);
    }

    x = this.layerNorm.call(
      x,
      getParam('layer_norm.weight', [this.dModel]),
      getParam('layer_norm.bias', [this.dModel]),
    );
    x = this.lmHead.call(x, getParam('lm_head.weight', [this.vocabSize, this.dModel]));
    return x;
  }
}

export class Whisper {
  public encoder: WhisperEncoder;
  public decoder: WhisperDecoder;

  constructor(
    dModel: number = 512,
    encoderAttentionHeads: number = 8,
    encoderFfnDim: number = 2048,
    encoderLayers: number = 6,
    decoderAttentionHeads: number = 8,
    decoderFfnDim: number = 2048,
    decoderLayers: number = 6,
    vocabSize: number = 51865,
  ) {
    this.encoder = new WhisperEncoder(dModel, encoderAttentionHeads, encoderFfnDim, encoderLayers);
    this.decoder = new WhisperDecoder(
      vocabSize,
      dModel,
      decoderAttentionHeads,
      decoderFfnDim,
      decoderLayers,
    );
  }

  call(inputFeatures: Tensor, decoderInputIds: Tensor): Tensor {
    const encoderHiddenStates = this.encoder.call(inputFeatures);
    return this.decoder.call(decoderInputIds, encoderHiddenStates);
  }
}

export function whisperTiny(): Whisper {
  return new Whisper(384, 6, 1536, 4, 6, 1536, 4, 51865);
}
