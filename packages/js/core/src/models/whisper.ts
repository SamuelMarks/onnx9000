/**
 * @fileoverview whisper.ts
 * Provides whisper functionality for the core package.
 */
import { Tensor } from '../ir/tensor.js';
import { recordOp } from '../macros.js';
import { ConvND, Gelu, Gemm, LayerNormalization, MultiHeadAttention } from '../primitives.js';

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

/** Represents the `WhisperEncoderLayer` class. */
export class WhisperEncoderLayer {
  /** Property `prefix` of `WhisperEncoderLayer`. */
  public prefix: string;
  /** Property `dModel` of `WhisperEncoderLayer`. */
  public dModel: number;
  /** Property `selfAttn` of `WhisperEncoderLayer`. */
  public selfAttn: MultiHeadAttention;
  /** Property `selfAttnLayerNorm` of `WhisperEncoderLayer`. */
  public selfAttnLayerNorm: LayerNormalization;
  /** Property `fc1` of `WhisperEncoderLayer`. */
  public fc1: Gemm;
  /** Property `act` of `WhisperEncoderLayer`. */
  public act: Gelu;
  /** Property `fc2` of `WhisperEncoderLayer`. */
  public fc2: Gemm;
  /** Property `finalLayerNorm` of `WhisperEncoderLayer`. */
  public finalLayerNorm: LayerNormalization;

  /**
   * Initializes a new `WhisperEncoderLayer` instance.
   * @param dModel Parameter `dModel`.
   * @param encoderAttentionHeads Parameter `encoderAttentionHeads`.
   * @param _encoderFfnDim Parameter `_encoderFfnDim`.
   * @param prefix Parameter `prefix`.
   */
  constructor(
    dModel: number,
    encoderAttentionHeads: number,
    _encoderFfnDim: number,
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

  /**
   * Executes `call` on `WhisperEncoderLayer`.
   * @param x Parameter `x`.
   * @returns Result of `call`.
   */
  call(x: Tensor): Tensor {
    let identity = x;
    let xNorm = this.selfAttnLayerNorm.call(
      x,
      getParam(`${this.prefix}.self_attn_layer_norm.weight`, [this.dModel]),
      getParam(`${this.prefix}.self_attn_layer_norm.bias`, [this.dModel]),
    );
    const xAttn = this.selfAttn.call(xNorm, xNorm, xNorm);
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

/** Represents the `WhisperEncoder` class. */
export class WhisperEncoder {
  /** Property `dModel` of `WhisperEncoder`. */
  public dModel: number;
  /** Property `conv1` of `WhisperEncoder`. */
  public conv1: ConvND;
  /** Property `act1` of `WhisperEncoder`. */
  public act1: Gelu;
  /** Property `conv2` of `WhisperEncoder`. */
  public conv2: ConvND;
  /** Property `act2` of `WhisperEncoder`. */
  public act2: Gelu;
  /** Property `layers` of `WhisperEncoder`. */
  public layers: WhisperEncoderLayer[];
  /** Property `layerNorm` of `WhisperEncoder`. */
  public layerNorm: LayerNormalization;

  /**
   * Initializes a new `WhisperEncoder` instance.
   * @param dModel Parameter `dModel`.
   * @param encoderAttentionHeads Parameter `encoderAttentionHeads`.
   * @param encoderFfnDim Parameter `encoderFfnDim`.
   * @param encoderLayers Parameter `encoderLayers`.
   */
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

  /**
   * Executes `call` on `WhisperEncoder`.
   * @param x Parameter `x`.
   * @returns Result of `call`.
   */
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

/** Represents the `WhisperDecoderLayer` class. */
export class WhisperDecoderLayer {
  /** Property `prefix` of `WhisperDecoderLayer`. */
  public prefix: string;
  /** Property `dModel` of `WhisperDecoderLayer`. */
  public dModel: number;
  /** Property `selfAttn` of `WhisperDecoderLayer`. */
  public selfAttn: MultiHeadAttention;
  /** Property `selfAttnLayerNorm` of `WhisperDecoderLayer`. */
  public selfAttnLayerNorm: LayerNormalization;
  /** Property `encoderAttn` of `WhisperDecoderLayer`. */
  public encoderAttn: MultiHeadAttention;
  /** Property `encoderAttnLayerNorm` of `WhisperDecoderLayer`. */
  public encoderAttnLayerNorm: LayerNormalization;
  /** Property `fc1` of `WhisperDecoderLayer`. */
  public fc1: Gemm;
  /** Property `act` of `WhisperDecoderLayer`. */
  public act: Gelu;
  /** Property `fc2` of `WhisperDecoderLayer`. */
  public fc2: Gemm;
  /** Property `finalLayerNorm` of `WhisperDecoderLayer`. */
  public finalLayerNorm: LayerNormalization;

  /**
   * Initializes a new `WhisperDecoderLayer` instance.
   * @param dModel Parameter `dModel`.
   * @param decoderAttentionHeads Parameter `decoderAttentionHeads`.
   * @param _decoderFfnDim Parameter `_decoderFfnDim`.
   * @param prefix Parameter `prefix`.
   */
  constructor(
    dModel: number,
    decoderAttentionHeads: number,
    _decoderFfnDim: number,
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

  /**
   * Executes `call` on `WhisperDecoderLayer`.
   * @param x Parameter `x`.
   * @param encoderHiddenStates Parameter `encoderHiddenStates`.
   * @param causalMask Parameter `causalMask`.
   * @returns Result of `call`.
   */
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

/** Represents the `WhisperDecoder` class. */
export class WhisperDecoder {
  /** Property `vocabSize` of `WhisperDecoder`. */
  public vocabSize: number;
  /** Property `dModel` of `WhisperDecoder`. */
  public dModel: number;
  /** Property `layers` of `WhisperDecoder`. */
  public layers: WhisperDecoderLayer[];
  /** Property `layerNorm` of `WhisperDecoder`. */
  public layerNorm: LayerNormalization;
  /** Property `lmHead` of `WhisperDecoder`. */
  public lmHead: Gemm;

  /**
   * Initializes a new `WhisperDecoder` instance.
   * @param vocabSize Parameter `vocabSize`.
   * @param dModel Parameter `dModel`.
   * @param decoderAttentionHeads Parameter `decoderAttentionHeads`.
   * @param decoderFfnDim Parameter `decoderFfnDim`.
   * @param decoderLayers Parameter `decoderLayers`.
   */
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

  /**
   * Executes `call` on `WhisperDecoder`.
   * @param inputIds Parameter `inputIds`.
   * @param encoderHiddenStates Parameter `encoderHiddenStates`.
   * @param causalMask Parameter `causalMask`.
   * @returns Result of `call`.
   */
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

/** Represents the `Whisper` class. */
export class Whisper {
  /** Property `encoder` of `Whisper`. */
  public encoder: WhisperEncoder;
  /** Property `decoder` of `Whisper`. */
  public decoder: WhisperDecoder;

  /**
   * Initializes a new `Whisper` instance.
   * @param dModel Parameter `dModel`.
   * @param encoderAttentionHeads Parameter `encoderAttentionHeads`.
   * @param encoderFfnDim Parameter `encoderFfnDim`.
   * @param encoderLayers Parameter `encoderLayers`.
   * @param decoderAttentionHeads Parameter `decoderAttentionHeads`.
   * @param decoderFfnDim Parameter `decoderFfnDim`.
   * @param decoderLayers Parameter `decoderLayers`.
   * @param vocabSize Parameter `vocabSize`.
   */
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

  /**
   * Executes `call` on `Whisper`.
   * @param inputFeatures Parameter `inputFeatures`.
   * @param decoderInputIds Parameter `decoderInputIds`.
   * @returns Result of `call`.
   */
  call(inputFeatures: Tensor, decoderInputIds: Tensor): Tensor {
    const encoderHiddenStates = this.encoder.call(inputFeatures);
    return this.decoder.call(decoderInputIds, encoderHiddenStates);
  }
}

/**
 * Executes `whisperTiny`.
 * @returns Result of `whisperTiny`.
 */
export function whisperTiny(): Whisper {
  return new Whisper(384, 6, 1536, 4, 6, 1536, 4, 51865);
}
