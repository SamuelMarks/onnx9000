/**
 * @fileoverview resnet.ts
 * Provides resnet functionality for the core package.
 */
import { Tensor } from '../ir/tensor.js';
import { recordOp } from '../macros.js';
import { BatchNormalization, ConvND, Gemm, Relu } from '../primitives.js';

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

/** Represents the `BasicBlock` class. */
export class BasicBlock {
  /** Property `expansion` of `BasicBlock`. */
  static expansion: number = 1;
  /** Property `prefix` of `BasicBlock`. */
  public prefix: string;
  /** Property `conv1` of `BasicBlock`. */
  public conv1: ConvND;
  /** Property `bn1` of `BasicBlock`. */
  public bn1: BatchNormalization;
  /** Property `relu` of `BasicBlock`. */
  public relu: Relu;
  /** Property `conv2` of `BasicBlock`. */
  public conv2: ConvND;
  /** Property `bn2` of `BasicBlock`. */
  public bn2: BatchNormalization;
  /** Property `downsample` of `BasicBlock`. */
  public downsample: boolean;
  /** Property `downsampleConv` of `BasicBlock`. */
  public downsampleConv?: ConvND;
  /** Property `downsampleBn` of `BasicBlock`. */
  public downsampleBn?: BatchNormalization;

  /**
   * Initializes a new `BasicBlock` instance.
   * @param inplanes Parameter `inplanes`.
   * @param planes Parameter `planes`.
   * @param stride Parameter `stride`.
   * @param downsample Parameter `downsample`.
   * @param prefix Parameter `prefix`.
   */
  constructor(
    inplanes: number,
    planes: number,
    stride: number = 1,
    downsample: boolean = false,
    prefix: string = '',
  ) {
    this.prefix = prefix;
    this.conv1 = new ConvND(2, inplanes, planes, 3, stride, 1, 1, 1, false);
    this.bn1 = new BatchNormalization(planes);
    this.relu = new Relu();
    this.conv2 = new ConvND(2, planes, planes, 3, 1, 1, 1, 1, false);
    this.bn2 = new BatchNormalization(planes);
    this.downsample = downsample;

    if (this.downsample) {
      this.downsampleConv = new ConvND(
        2,
        inplanes,
        planes * BasicBlock.expansion,
        1,
        stride,
        0,
        1,
        1,
        false,
      );
      this.downsampleBn = new BatchNormalization(planes * BasicBlock.expansion);
    }
  }

  /**
   * Executes `call` on `BasicBlock`.
   * @param x Parameter `x`.
   * @returns Result of `call`.
   */
  call(x: Tensor): Tensor {
    let identity = x;

    let out = this.conv1.call(
      x,
      getParam(`${this.prefix}.conv1.weight`, [
        this.conv1.outChannels,
        this.conv1.inChannels,
        3,
        3,
      ]),
    );
    out = this.bn1.call(
      out,
      getParam(`${this.prefix}.bn1.weight`, [this.bn1.numFeatures]),
      getParam(`${this.prefix}.bn1.bias`, [this.bn1.numFeatures]),
      getParam(`${this.prefix}.bn1.running_mean`, [this.bn1.numFeatures]),
      getParam(`${this.prefix}.bn1.running_var`, [this.bn1.numFeatures]),
    );
    out = this.relu.call(out);

    out = this.conv2.call(
      out,
      getParam(`${this.prefix}.conv2.weight`, [
        this.conv2.outChannels,
        this.conv2.inChannels,
        3,
        3,
      ]),
    );
    out = this.bn2.call(
      out,
      getParam(`${this.prefix}.bn2.weight`, [this.bn2.numFeatures]),
      getParam(`${this.prefix}.bn2.bias`, [this.bn2.numFeatures]),
      getParam(`${this.prefix}.bn2.running_mean`, [this.bn2.numFeatures]),
      getParam(`${this.prefix}.bn2.running_var`, [this.bn2.numFeatures]),
    );

    if (this.downsample && this.downsampleConv && this.downsampleBn) {
      identity = this.downsampleConv.call(
        x,
        getParam(`${this.prefix}.downsample.0.weight`, [
          this.downsampleConv.outChannels,
          this.downsampleConv.inChannels,
          1,
          1,
        ]),
      );
      identity = this.downsampleBn.call(
        identity,
        getParam(`${this.prefix}.downsample.1.weight`, [this.downsampleBn.numFeatures]),
        getParam(`${this.prefix}.downsample.1.bias`, [this.downsampleBn.numFeatures]),
        getParam(`${this.prefix}.downsample.1.running_mean`, [this.downsampleBn.numFeatures]),
        getParam(`${this.prefix}.downsample.1.running_var`, [this.downsampleBn.numFeatures]),
      );
    }

    out = recordOp('Add', [out, identity]);
    out = this.relu.call(out);

    return out;
  }
}

/** Represents the `ResNet` class. */
export class ResNet {
  /** Property `inplanes` of `ResNet`. */
  public inplanes: number = 64;
  /** Property `conv1` of `ResNet`. */
  public conv1: ConvND;
  /** Property `bn1` of `ResNet`. */
  public bn1: BatchNormalization;
  /** Property `relu` of `ResNet`. */
  public relu: Relu;
  /** Property `layer1` of `ResNet`. */
  public layer1: BasicBlock[];
  /** Property `layer2` of `ResNet`. */
  public layer2: BasicBlock[];
  /** Property `layer3` of `ResNet`. */
  public layer3: BasicBlock[];
  /** Property `layer4` of `ResNet`. */
  public layer4: BasicBlock[];
  /** Property `fc` of `ResNet`. */
  public fc: Gemm;
  /** Property `numClasses` of `ResNet`. */
  public numClasses: number;

  /**
   * Initializes a new `ResNet` instance.
   * @param layers Parameter `layers`.
   * @param numClasses Parameter `numClasses`.
   */
  constructor(layers: number[], numClasses: number = 1000) {
    this.conv1 = new ConvND(2, 3, this.inplanes, 7, 2, 3, 1, 1, false);
    this.bn1 = new BatchNormalization(this.inplanes);
    this.relu = new Relu();

    this.layer1 = this.makeLayer(64, layers[0] as number, 1, 'layer1');
    this.layer2 = this.makeLayer(128, layers[1] as number, 2, 'layer2');
    this.layer3 = this.makeLayer(256, layers[2] as number, 2, 'layer3');
    this.layer4 = this.makeLayer(512, layers[3] as number, 2, 'layer4');

    this.fc = new Gemm(1.0, 1.0, 0, 1); // transB = 1
    this.numClasses = numClasses;
  }

  /**
   * Executes `makeLayer` on `ResNet`.
   * @param planes Parameter `planes`.
   * @param blocks Parameter `blocks`.
   * @param stride Parameter `stride`.
   * @param prefix Parameter `prefix`.
   * @returns Result of `makeLayer`.
   */
  private makeLayer(
    planes: number,
    blocks: number,
    stride: number = 1,
    prefix: string = '',
  ): BasicBlock[] {
    let downsample = false;
    if (stride !== 1 || this.inplanes !== planes * BasicBlock.expansion) {
      downsample = true;
    }

    const layers: BasicBlock[] = [];
    layers.push(new BasicBlock(this.inplanes, planes, stride, downsample, `${prefix}.0`));
    this.inplanes = planes * BasicBlock.expansion;
    for (let i = 1; i < blocks; i++) {
      layers.push(new BasicBlock(this.inplanes, planes, 1, false, `${prefix}.${i}`));
    }

    return layers;
  }

  /**
   * Executes `call` on `ResNet`.
   * @param x Parameter `x`.
   * @returns Result of `call`.
   */
  call(x: Tensor): Tensor {
    x = this.conv1.call(
      x,
      getParam('conv1.weight', [this.conv1.outChannels, this.conv1.inChannels, 7, 7]),
    );
    x = this.bn1.call(
      x,
      getParam('bn1.weight', [this.bn1.numFeatures]),
      getParam('bn1.bias', [this.bn1.numFeatures]),
      getParam('bn1.running_mean', [this.bn1.numFeatures]),
      getParam('bn1.running_var', [this.bn1.numFeatures]),
    );
    x = this.relu.call(x);

    x = recordOp('MaxPool', [x], {
      kernel_shape: [3, 3],
      strides: [2, 2],
      pads: [1, 1, 1, 1],
    });

    for (const layer of [this.layer1, this.layer2, this.layer3, this.layer4]) {
      for (const block of layer) {
        x = block.call(x);
      }
    }

    x = recordOp('GlobalAveragePool', [x]);
    x = recordOp('Flatten', [x]);

    const fcW = getParam('fc.weight', [this.numClasses, 512 * BasicBlock.expansion]);
    const fcB = getParam('fc.bias', [this.numClasses]);

    x = this.fc.call(x, fcW, fcB);

    return x;
  }
}

/**
 * Executes `resnet18`.
 * @param numClasses Parameter `numClasses`.
 * @returns Result of `resnet18`.
 */
export function resnet18(numClasses: number = 1000): ResNet {
  return new ResNet([2, 2, 2, 2], numClasses);
}

/**
 * Executes `resnet50`.
 * @param numClasses Parameter `numClasses`.
 * @returns Result of `resnet50`.
 */
export function resnet50(numClasses: number = 1000): ResNet {
  return new ResNet([3, 4, 6, 3], numClasses);
}
