/**
 * @fileoverview convnext.ts
 * Provides convnext functionality for the core package.
 */
import { Tensor } from '../ir/tensor.js';
import { recordOp } from '../macros.js';
import { ConvND, DepthwiseConv, Gelu, Gemm, LayerNormalization } from '../primitives.js';

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

/** Represents the `ConvNeXtBlock` class. */
export class ConvNeXtBlock {
  /** Property `prefix` of `ConvNeXtBlock`. */
  public prefix: string;
  /** Property `dim` of `ConvNeXtBlock`. */
  public dim: number;
  /** Property `dwconv` of `ConvNeXtBlock`. */
  public dwconv: DepthwiseConv;
  /** Property `norm` of `ConvNeXtBlock`. */
  public norm: LayerNormalization;
  /** Property `pwconv1` of `ConvNeXtBlock`. */
  public pwconv1: ConvND;
  /** Property `act` of `ConvNeXtBlock`. */
  public act: Gelu;
  /** Property `pwconv2` of `ConvNeXtBlock`. */
  public pwconv2: ConvND;

  /**
   * Initializes a new `ConvNeXtBlock` instance.
   * @param dim Parameter `dim`.
   * @param _dropPath Parameter `_dropPath`.
   * @param prefix Parameter `prefix`.
   */
  constructor(dim: number, _dropPath: number = 0.0, prefix: string = '') {
    this.prefix = prefix;
    this.dim = dim;
    this.dwconv = new DepthwiseConv(2, dim, 7, 1, 3, 1, false);
    this.norm = new LayerNormalization([dim], 1e-6);
    this.pwconv1 = new ConvND(2, dim, 4 * dim, 1, 1, 0, 1, 1, false);
    this.act = new Gelu();
    this.pwconv2 = new ConvND(2, 4 * dim, dim, 1, 1, 0, 1, 1, false);
  }

  /**
   * Executes `call` on `ConvNeXtBlock`.
   * @param x Parameter `x`.
   * @returns Result of `call`.
   */
  call(x: Tensor): Tensor {
    const identity = x;

    let out = this.dwconv.call(
      x,
      getParam(`${this.prefix}.dwconv.weight`, [this.dwconv.outChannels, 1, 7, 7]),
      getParam(`${this.prefix}.dwconv.bias`, [this.dwconv.outChannels]),
    );

    out = this.norm.call(
      out,
      getParam(`${this.prefix}.norm.weight`, [this.dim]),
      getParam(`${this.prefix}.norm.bias`, [this.dim]),
    );

    out = this.pwconv1.call(
      out,
      getParam(`${this.prefix}.pwconv1.weight`, [
        this.pwconv1.outChannels,
        this.pwconv1.inChannels,
        1,
        1,
      ]),
      getParam(`${this.prefix}.pwconv1.bias`, [this.pwconv1.outChannels]),
    );
    out = this.act.call(out);

    out = this.pwconv2.call(
      out,
      getParam(`${this.prefix}.pwconv2.weight`, [
        this.pwconv2.outChannels,
        this.pwconv2.inChannels,
        1,
        1,
      ]),
      getParam(`${this.prefix}.pwconv2.bias`, [this.pwconv2.outChannels]),
    );

    return recordOp('Add', [identity, out]);
  }
}

/** Represents the `ConvNeXt` class. */
export class ConvNeXt {
  /** Property `numClasses` of `ConvNeXt`. */
  public numClasses: number;
  /** Property `stemConv` of `ConvNeXt`. */
  public stemConv: ConvND;
  /** Property `stemNorm` of `ConvNeXt`. */
  public stemNorm: LayerNormalization;
  /** Property `block1` of `ConvNeXt`. */
  public block1: ConvNeXtBlock;
  /** Property `block2` of `ConvNeXt`. */
  public block2: ConvNeXtBlock;
  /** Property `headNorm` of `ConvNeXt`. */
  public headNorm: LayerNormalization;
  /** Property `head` of `ConvNeXt`. */
  public head: Gemm;

  /**
   * Initializes a new `ConvNeXt` instance.
   * @param inChans Parameter `inChans`.
   * @param numClasses Parameter `numClasses`.
   */
  constructor(inChans: number = 3, numClasses: number = 1000) {
    this.numClasses = numClasses;
    this.stemConv = new ConvND(2, inChans, 96, 4, 4, 0, 1, 1, false);
    this.stemNorm = new LayerNormalization([96], 1e-6);

    this.block1 = new ConvNeXtBlock(96, 0.0, 'block1');
    this.block2 = new ConvNeXtBlock(96, 0.0, 'block2');

    this.headNorm = new LayerNormalization([96], 1e-6);
    this.head = new Gemm(1.0, 1.0, 0, 1);
  }

  /**
   * Executes `call` on `ConvNeXt`.
   * @param x Parameter `x`.
   * @returns Result of `call`.
   */
  call(x: Tensor): Tensor {
    x = this.stemConv.call(
      x,
      getParam('stem_conv.weight', [this.stemConv.outChannels, this.stemConv.inChannels, 4, 4]),
      getParam('stem_conv.bias', [this.stemConv.outChannels]),
    );
    x = this.stemNorm.call(x, getParam('stem_norm.weight', [96]), getParam('stem_norm.bias', [96]));

    x = this.block1.call(x);
    x = this.block2.call(x);

    x = recordOp('GlobalAveragePool', [x]);
    x = recordOp('Flatten', [x]);

    x = this.headNorm.call(x, getParam('head_norm.weight', [96]), getParam('head_norm.bias', [96]));
    x = this.head.call(
      x,
      getParam('head.weight', [this.numClasses, 96]),
      getParam('head.bias', [this.numClasses]),
    );

    return x;
  }
}

/**
 * Executes `convnextTiny`.
 * @param numClasses Parameter `numClasses`.
 * @returns Result of `convnextTiny`.
 */
export function convnextTiny(numClasses: number = 1000): ConvNeXt {
  return new ConvNeXt(3, numClasses);
}
