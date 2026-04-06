import { Tensor } from '../ir/tensor.js';
import { BatchNormalization, ConvND, DepthwiseConv, Gemm, Sigmoid, Silu } from '../primitives.js';

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

export class SqueezeExcitation {
  public prefix: string;
  public inChannels: number;
  public squeezeChannels: number;
  public fc1: Gemm;
  public act: Silu;
  public fc2: Gemm;
  public scaleAct: Sigmoid;

  constructor(inChannels: number, squeezeChannels: number, prefix: string = '') {
    this.prefix = prefix;
    this.inChannels = inChannels;
    this.squeezeChannels = squeezeChannels;
    this.fc1 = new Gemm(1.0, 1.0, 0, 1);
    this.act = new Silu();
    this.fc2 = new Gemm(1.0, 1.0, 0, 1);
    this.scaleAct = new Sigmoid();
  }

  call(x: Tensor): Tensor {
    let scale = recordOp('GlobalAveragePool', [x]);
    scale = recordOp('Flatten', [scale]);

    const fc1W = getParam(`${this.prefix}.fc1.weight`, [this.squeezeChannels, this.inChannels]);
    const fc1B = getParam(`${this.prefix}.fc1.bias`, [this.squeezeChannels]);
    scale = this.fc1.call(scale, fc1W, fc1B);
    scale = this.act.call(scale);

    const fc2W = getParam(`${this.prefix}.fc2.weight`, [this.inChannels, this.squeezeChannels]);
    const fc2B = getParam(`${this.prefix}.fc2.bias`, [this.inChannels]);
    scale = this.fc2.call(scale, fc2W, fc2B);
    scale = this.scaleAct.call(scale);

    const shapeTensor = recordOp('Constant', [], { value: [-1, this.inChannels, 1, 1] });
    scale = recordOp('Reshape', [scale, shapeTensor]);

    return recordOp('Mul', [x, scale]);
  }
}

export class MBConv {
  public prefix: string;
  public inChannels: number;
  public outChannels: number;
  public expandRatio: number;
  public stride: number;
  public useResConnect: boolean;

  public expandConv?: ConvND;
  public bn0?: BatchNormalization;
  public act0?: Silu;

  public depthwiseConv: DepthwiseConv;
  public bn1: BatchNormalization;
  public act1: Silu;
  public se: SqueezeExcitation;
  public projectConv: ConvND;
  public bn2: BatchNormalization;

  constructor(
    inChannels: number,
    outChannels: number,
    expandRatio: number,
    stride: number,
    kernelSize: number,
    prefix: string = '',
  ) {
    this.prefix = prefix;
    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.stride = stride;
    this.useResConnect = this.stride === 1 && inChannels === outChannels;
    this.expandRatio = expandRatio;

    const hiddenDim = inChannels * expandRatio;

    if (expandRatio !== 1) {
      this.expandConv = new ConvND(2, inChannels, hiddenDim, 1, 1, 0, 1, 1, false);
      this.bn0 = new BatchNormalization(hiddenDim);
      this.act0 = new Silu();
    }

    this.depthwiseConv = new DepthwiseConv(
      2,
      hiddenDim,
      kernelSize,
      stride,
      Math.floor(kernelSize / 2),
      1,
      false,
    );
    this.bn1 = new BatchNormalization(hiddenDim);
    this.act1 = new Silu();

    this.se = new SqueezeExcitation(
      hiddenDim,
      Math.max(1, Math.floor(inChannels / 4)),
      `${prefix}.se`,
    );

    this.projectConv = new ConvND(2, hiddenDim, outChannels, 1, 1, 0, 1, 1, false);
    this.bn2 = new BatchNormalization(outChannels);
  }

  call(x: Tensor): Tensor {
    let identity = x;

    if (this.expandRatio !== 1 && this.expandConv && this.bn0 && this.act0) {
      x = this.expandConv.call(
        x,
        getParam(`${this.prefix}.expand_conv.weight`, [
          this.expandConv.outChannels,
          this.expandConv.inChannels,
          1,
          1,
        ]),
      );
      x = this.bn0.call(
        x,
        getParam(`${this.prefix}.bn0.weight`, [this.bn0.numFeatures]),
        getParam(`${this.prefix}.bn0.bias`, [this.bn0.numFeatures]),
        getParam(`${this.prefix}.bn0.running_mean`, [this.bn0.numFeatures]),
        getParam(`${this.prefix}.bn0.running_var`, [this.bn0.numFeatures]),
      );
      x = this.act0.call(x);
    }

    const ks = Array.isArray(this.depthwiseConv.kernelSize)
      ? this.depthwiseConv.kernelSize[0]
      : this.depthwiseConv.kernelSize;
    x = this.depthwiseConv.call(
      x,
      getParam(`${this.prefix}.depthwise_conv.weight`, [
        this.depthwiseConv.outChannels!,
        1,
        ks!,
        ks!,
      ]),
    );
    x = this.bn1.call(
      x,
      getParam(`${this.prefix}.bn1.weight`, [this.bn1.numFeatures]),
      getParam(`${this.prefix}.bn1.bias`, [this.bn1.numFeatures]),
      getParam(`${this.prefix}.bn1.running_mean`, [this.bn1.numFeatures]),
      getParam(`${this.prefix}.bn1.running_var`, [this.bn1.numFeatures]),
    );
    x = this.act1.call(x);

    x = this.se.call(x);

    x = this.projectConv.call(
      x,
      getParam(`${this.prefix}.project_conv.weight`, [
        this.projectConv.outChannels,
        this.projectConv.inChannels,
        1,
        1,
      ]),
    );
    x = this.bn2.call(
      x,
      getParam(`${this.prefix}.bn2.weight`, [this.bn2.numFeatures]),
      getParam(`${this.prefix}.bn2.bias`, [this.bn2.numFeatures]),
      getParam(`${this.prefix}.bn2.running_mean`, [this.bn2.numFeatures]),
      getParam(`${this.prefix}.bn2.running_var`, [this.bn2.numFeatures]),
    );

    if (this.useResConnect) {
      x = recordOp('Add', [identity, x]);
    }

    return x;
  }
}

export class EfficientNet {
  public numClasses: number;
  public stemConv: ConvND;
  public stemBn: BatchNormalization;
  public stemAct: Silu;
  public block1: MBConv;
  public block2: MBConv;
  public headConv: ConvND;
  public headBn: BatchNormalization;
  public headAct: Silu;
  public classifier: Gemm;

  constructor(numClasses: number = 1000) {
    this.numClasses = numClasses;
    this.stemConv = new ConvND(2, 3, 32, 3, 2, 1, 1, 1, false);
    this.stemBn = new BatchNormalization(32);
    this.stemAct = new Silu();

    this.block1 = new MBConv(32, 16, 1, 1, 3, 'block1');
    this.block2 = new MBConv(16, 24, 6, 2, 3, 'block2');

    this.headConv = new ConvND(2, 24, 1280, 1, 1, 0, 1, 1, false);
    this.headBn = new BatchNormalization(1280);
    this.headAct = new Silu();

    this.classifier = new Gemm(1.0, 1.0, 0, 1);
  }

  call(x: Tensor): Tensor {
    x = this.stemConv.call(
      x,
      getParam('stem_conv.weight', [this.stemConv.outChannels, this.stemConv.inChannels, 3, 3]),
    );
    x = this.stemBn.call(
      x,
      getParam('stem_bn.weight', [this.stemBn.numFeatures]),
      getParam('stem_bn.bias', [this.stemBn.numFeatures]),
      getParam('stem_bn.running_mean', [this.stemBn.numFeatures]),
      getParam('stem_bn.running_var', [this.stemBn.numFeatures]),
    );
    x = this.stemAct.call(x);

    x = this.block1.call(x);
    x = this.block2.call(x);

    x = this.headConv.call(
      x,
      getParam('head_conv.weight', [this.headConv.outChannels, this.headConv.inChannels, 1, 1]),
    );
    x = this.headBn.call(
      x,
      getParam('head_bn.weight', [this.headBn.numFeatures]),
      getParam('head_bn.bias', [this.headBn.numFeatures]),
      getParam('head_bn.running_mean', [this.headBn.numFeatures]),
      getParam('head_bn.running_var', [this.headBn.numFeatures]),
    );
    x = this.headAct.call(x);

    x = recordOp('GlobalAveragePool', [x]);
    x = recordOp('Flatten', [x]);
    x = this.classifier.call(
      x,
      getParam('classifier.weight', [this.numClasses, 1280]),
      getParam('classifier.bias', [this.numClasses]),
    );

    return x;
  }
}

export function efficientnetB0(numClasses: number = 1000): EfficientNet {
  return new EfficientNet(numClasses);
}
