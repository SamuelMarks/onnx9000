import { describe, expect, it } from 'vitest';
import * as index from '../src/index.js';
import * as caffe from '../src/mmdnn/caffe/index.js';
import * as cntk from '../src/mmdnn/cntk/index.js';
import * as coreml from '../src/mmdnn/coreml/index.js';
import * as darknet from '../src/mmdnn/darknet/index.js';
import * as mmdnn from '../src/mmdnn/index.js';
import * as keras from '../src/mmdnn/keras/index.js';
import * as legacy from '../src/mmdnn/legacy/index.js';
import * as mxnet from '../src/mmdnn/mxnet/index.js';
import * as ncnn from '../src/mmdnn/ncnn/index.js';
import * as paddle from '../src/mmdnn/paddle/index.js';
import * as pytorch from '../src/mmdnn/pytorch/index.js';
import * as tensorflow from '../src/mmdnn/tensorflow/index.js';
import * as tfjs from '../src/mmdnn/tfjs/index.js';
import * as verification from '../src/mmdnn/verification/index.js';

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
