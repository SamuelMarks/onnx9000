import { describe, it, expect } from 'vitest';
import * as index from '../src/index';
import * as mmdnn from '../src/mmdnn/index';
import * as caffe from '../src/mmdnn/caffe/index';
import * as mxnet from '../src/mmdnn/mxnet/index';
import * as cntk from '../src/mmdnn/cntk/index';
import * as pytorch from '../src/mmdnn/pytorch/index';
import * as tfjs from '../src/mmdnn/tfjs/index';
import * as tensorflow from '../src/mmdnn/tensorflow/index';
import * as coreml from '../src/mmdnn/coreml/index';
import * as darknet from '../src/mmdnn/darknet/index';
import * as ncnn from '../src/mmdnn/ncnn/index';
import * as paddle from '../src/mmdnn/paddle/index';
import * as verification from '../src/mmdnn/verification/index';
import * as legacy from '../src/mmdnn/legacy/index';
import * as keras from '../src/mmdnn/keras/index';

describe('Barrel file coverage', () => {
  it('should export expected symbols from main index', () => {
    expect(index.mmdnn).toBeDefined();
    expect(index.keras2onnx).toBeDefined();
  });

  it('should export expected symbols from submodules', () => {
    expect(mmdnn.convert).toBeDefined();
    expect(caffe.parsePrototxt).toBeDefined();
    expect(mxnet.parseMxNetSymbol).toBeDefined();
    expect(cntk.CNTKParser).toBeDefined();
    expect(pytorch.PyTorchGenerator).toBeDefined();
    expect(tfjs.generateTFJSCode).toBeDefined();
    expect(tensorflow.parsePbtxt).toBeDefined();
    expect(coreml.CoreMLImporter).toBeDefined();
    expect(darknet.parseCfg).toBeDefined();
    expect(ncnn.NcnnMapper).toBeDefined();
    expect(paddle.PaddleParser).toBeDefined();
    expect(verification.ONNXNormalizer).toBeDefined();
    expect(legacy.LegacyQuirkResolver).toBeDefined();
    expect(keras.KerasGenerator).toBeDefined();
    expect(keras.KerasImporter).toBeDefined();
  });
});
