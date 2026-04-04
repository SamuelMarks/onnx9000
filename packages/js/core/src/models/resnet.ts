import { Tensor } from '../ir/tensor.js';
import { recordOp } from '../macros.js';
const add = (a: Tensor, b: Tensor) => recordOp('Add', [a, b]);
const globalAveragePool = (x: Tensor) => recordOp('GlobalAveragePool', [x]);
const maxPool = (x: Tensor, attr?: any) => recordOp('MaxPool', [x], attr);
const flatten = (x: Tensor) => recordOp('Flatten', [x]);

import { BatchNormalization, ConvND, Gemm, Relu } from '../primitives.js';

function getParam(name: string, shape: number[], dtype: any = 'float32'): Tensor {
  return new Tensor(name, shape, dtype, false, false, new Float32Array());
}

export class BasicBlock {
  static expansion: number = 1;
  public prefix: string;
  public conv1: ConvND;
  public bn1: BatchNormalization;
  public relu: Relu;
  public conv2: ConvND;
  public bn2: BatchNormalization;
  public downsample: boolean;
  public downsampleConv?: ConvND;
  public downsampleBn?: BatchNormalization;

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

    // We use a dummy add operation that returns a generic Tensor for now
    // Assuming add is in registry or primitives
    // Let's implement dummy fallback
    const recordOp = (opType: string, inputs: Tensor[], attr?: any) => {
      const dtype = inputs[0]?.dtype ?? 'float32';
      return new Tensor(`${opType}_out`, [], dtype, false, false, new Float32Array());
    };

    out = recordOp('Add', [out, identity]);
    out = this.relu.call(out);

    return out;
  }
}

export class ResNet {
  public inplanes: number = 64;
  public conv1: ConvND;
  public bn1: BatchNormalization;
  public relu: Relu;
  public layer1: BasicBlock[];
  public layer2: BasicBlock[];
  public layer3: BasicBlock[];
  public layer4: BasicBlock[];
  public fc: Gemm;
  public numClasses: number;

  constructor(layers: number[], numClasses: number = 1000) {
    this.conv1 = new ConvND(2, 3, this.inplanes, 7, 2, 3, 1, 1, false);
    this.bn1 = new BatchNormalization(this.inplanes);
    this.relu = new Relu();

    this.layer1 = this.makeLayer(64, layers[0]!, 1, 'layer1');
    this.layer2 = this.makeLayer(128, layers[1]!, 2, 'layer2');
    this.layer3 = this.makeLayer(256, layers[2]!, 2, 'layer3');
    this.layer4 = this.makeLayer(512, layers[3]!, 2, 'layer4');

    this.fc = new Gemm(1.0, 1.0, 0, 1); // transB = 1
    this.numClasses = numClasses;
  }

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

    const recordOp = (opType: string, inputs: Tensor[], attr?: any) => {
      const dtype = inputs[0]?.dtype ?? 'float32';
      return new Tensor(`${opType}_out`, [], dtype, false, false, new Float32Array());
    };

    x = recordOp('MaxPool', [x], { kernel_shape: [3, 3], strides: [2, 2], pads: [1, 1, 1, 1] });

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

export function resnet18(numClasses: number = 1000): ResNet {
  return new ResNet([2, 2, 2, 2], numClasses);
}

export function resnet50(numClasses: number = 1000): ResNet {
  return new ResNet([3, 4, 6, 3], numClasses);
}
