/* eslint-disable */
// @ts-nocheck
import { FileNode } from '../components/FileTree';

export interface ExampleData {
  id: string;
  label: string;
  root: FileNode;
  initialFile?: string;
}

export const LHS_FRAMEWORKS = [
  { value: 'keras', label: 'Keras' },
  { value: 'onnxscript', label: 'onnxscript / spox' },
  { value: 'tensorflow', label: 'TensorFlow' },
  { value: 'caffe', label: 'Caffe' },
  { value: 'mxnet', label: 'MXNet' },
  { value: 'paddle', label: 'PaddlePaddle' },
  { value: 'scikitlearn', label: 'Scikit-Learn' },
  { value: 'lightgbm', label: 'LightGBM' },
  { value: 'xgboost', label: 'XGBoost' },
  { value: 'catboost', label: 'CatBoost' },
  { value: 'sparkml', label: 'SparkML' }
];

export const LHS_EXAMPLES: Record<string, ExampleData[]> = {
  keras: [
    {
      id: 'keras-mnist',
      label: 'Basic CNN (MNIST)',
      initialFile: '/keras-mnist/train.py',
      root: {
        name: 'keras-mnist',
        type: 'directory',
        path: '/keras-mnist',
        children: [
          { name: 'model.h5', type: 'file', path: '/keras-mnist/model.h5' },
          {
            name: 'model.json',
            type: 'file',
            path: '/keras-mnist/model.json',
            content: JSON.stringify(
              {
                format: 'layers-model',
                modelTopology: {
                  class_name: 'Sequential',
                  config: {
                    name: 'mnist_model',
                    layers: [
                      {
                        class_name: 'InputLayer',
                        config: { name: 'image_input', batch_input_shape: [null, 28, 28, 1] }
                      },
                      {
                        class_name: 'Conv2D',
                        config: { name: 'conv2d_1', filters: 32, kernel_size: [3, 3] }
                      },
                      {
                        class_name: 'MaxPooling2D',
                        config: { name: 'max_pooling2d', pool_size: [2, 2] }
                      },
                      {
                        class_name: 'Flatten',
                        config: { name: 'flatten' }
                      },
                      {
                        class_name: 'Dense',
                        config: { name: 'dense_1', units: 128 }
                      },
                      {
                        class_name: 'Dense',
                        config: { name: 'output_probs', units: 10 }
                      }
                    ]
                  }
                },
                weightsManifest: []
              },
              null,
              2
            )
          },
          {
            name: 'train.py',
            type: 'file',
            path: '/keras-mnist/train.py',
            content: `import keras
from keras import layers, models

# A simple sequential neural network
model = models.Sequential([
    layers.Input(shape=(28, 28, 1), name='image_input'),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax', name='output_probs')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model ...
# model.fit(x_train, y_train, epochs=5)

# Export to HDF5
model.save('model.h5')
`
          }
        ]
      }
    },
    {
      id: 'keras-resnet',
      label: 'ResNet50 (ImageNet)',
      initialFile: '/keras-resnet/train_resnet.py',
      root: {
        name: 'keras-resnet',
        type: 'directory',
        path: '/keras-resnet',
        children: [
          { name: 'resnet50.h5', type: 'file', path: '/keras-resnet/resnet50.h5' },
          { name: 'train_resnet.py', type: 'file', path: '/keras-resnet/train_resnet.py' },
          { name: 'imagenet_classes.txt', type: 'file', path: '/keras-resnet/imagenet_classes.txt' }
        ]
      }
    }
  ],
  onnxscript: [
    {
      id: 'spox-mlp',
      label: 'Simple MLP (Spox)',
      initialFile: '/spox-mlp/model.py',
      root: {
        name: 'spox-mlp',
        type: 'directory',
        path: '/spox-mlp',
        children: [
          {
            name: 'model.py',
            type: 'file',
            path: '/spox-mlp/model.py',
            content: `import onnxscript
from onnxscript import opset15 as op
from onnxscript import FLOAT

@onnxscript.script()
def mlp(X: FLOAT[10, 10], W1: FLOAT[10, 20], B1: FLOAT[20], W2: FLOAT[20, 1], B2: FLOAT[1]) -> FLOAT[10, 1]:
    H1 = op.MatMul(X, W1)
    H1_bias = op.Add(H1, B1)
    H1_relu = op.Relu(H1_bias)
    H2 = op.MatMul(H1_relu, W2)
    Y = op.Add(H2, B2)
    return Y
`
          }
        ]
      }
    }
  ],
  tensorflow: [
    {
      id: 'tf-mobilenet',
      label: 'MobileNet V2',
      initialFile: '/tf-mobilenet/saved_model.pbtxt',
      root: {
        name: 'tf-mobilenet',
        type: 'directory',
        path: '/tf-mobilenet',
        children: [
          {
            name: 'saved_model.pbtxt',
            type: 'file',
            path: '/tf-mobilenet/saved_model.pbtxt',
            content: `node {
  name: "input_1"
  op: "Placeholder"
  attr {
    key: "dtype"
    value { type: DT_FLOAT }
  }
  attr {
    key: "shape"
    value { shape { dim { size: -1 } dim { size: 224 } dim { size: 224 } dim { size: 3 } } }
  }
}
node {
  name: "weights"
  op: "Const"
  attr {
    key: "value"
    value { tensor { dtype: DT_FLOAT tensor_shape { dim { size: 32 } dim { size: 3 } dim { size: 3 } dim { size: 3 } } } }
  }
}
node {
  name: "Conv2D_1"
  op: "Conv2D"
  input: "input_1"
  input: "weights"
  attr {
    key: "strides"
    value { list { i: 1 i: 2 i: 2 i: 1 } }
  }
  attr {
    key: "padding"
    value { s: "SAME" }
  }
}
node {
  name: "Relu_1"
  op: "Relu"
  input: "Conv2D_1"
}`
          },
          { name: 'export.py', type: 'file', path: '/tf-mobilenet/export.py' },
          {
            name: 'variables',
            type: 'directory',
            path: '/tf-mobilenet/variables',
            children: [
              {
                name: 'variables.data-00000-of-00001',
                type: 'file',
                path: '/tf-mobilenet/variables/variables.data-00000-of-00001'
              },
              {
                name: 'variables.index',
                type: 'file',
                path: '/tf-mobilenet/variables/variables.index'
              }
            ]
          }
        ]
      }
    },
    {
      id: 'tf-bert',
      label: 'BERT Base (NLP)',
      initialFile: '/tf-bert/inference.py',
      root: {
        name: 'tf-bert',
        type: 'directory',
        path: '/tf-bert',
        children: [
          { name: 'saved_model.pb', type: 'file', path: '/tf-bert/saved_model.pb' },
          { name: 'inference.py', type: 'file', path: '/tf-bert/inference.py' },
          { name: 'vocab.txt', type: 'file', path: '/tf-bert/vocab.txt' }
        ]
      }
    }
  ],
  caffe: [
    {
      id: 'caffe-alexnet',
      label: 'AlexNet',
      initialFile: '/caffe-alexnet/deploy.prototxt',
      root: {
        name: 'caffe-alexnet',
        type: 'directory',
        path: '/caffe-alexnet',
        children: [
          {
            name: 'deploy.prototxt',
            type: 'file',
            path: '/caffe-alexnet/deploy.prototxt',
            content:
              'name: "AlexNet"\nlayer {\n  name: "data"\n  type: "Input"\n  top: "data"\n  input_param {\n    shape {\n      dim: 1\n      dim: 3\n      dim: 227\n      dim: 227\n    }\n  }\n}\nlayer {\n  name: "conv1"\n  type: "Convolution"\n  bottom: "data"\n  top: "conv1"\n  convolution_param {\n    num_output: 96\n    kernel_size: 11\n    stride: 4\n  }\n}\nlayer {\n  name: "relu1"\n  type: "ReLU"\n  bottom: "conv1"\n  top: "conv1"\n}\nlayer {\n  name: "pool1"\n  type: "Pooling"\n  bottom: "conv1"\n  top: "pool1"\n  pooling_param {\n    pool: MAX\n    kernel_size: 3\n    stride: 2\n  }\n}\nlayer {\n  name: "fc8"\n  type: "InnerProduct"\n  bottom: "pool1"\n  top: "fc8"\n  inner_product_param {\n    num_output: 1000\n  }\n}\nlayer {\n  name: "prob"\n  type: "Softmax"\n  bottom: "fc8"\n  top: "prob"\n}'
          },
          { name: 'weights.caffemodel', type: 'file', path: '/caffe-alexnet/weights.caffemodel' }
        ]
      }
    }
  ],
  mxnet: [
    {
      id: 'mxnet-resnet50',
      label: 'ResNet-50',
      initialFile: '/mxnet-resnet50/model-symbol.json',
      root: {
        name: 'mxnet-resnet50',
        type: 'directory',
        path: '/mxnet-resnet50',
        children: [
          {
            name: 'model-symbol.json',
            type: 'file',
            path: '/mxnet-resnet50/model-symbol.json',
            content:
              '{"nodes": [{"op": "null", "name": "data", "inputs": []}, {"op": "Convolution", "name": "conv0", "attrs": {"kernel": "(7, 7)", "num_filter": "64", "stride": "(2, 2)", "pad": "(3, 3)", "no_bias": "True"}, "inputs": [[0, 0, 0], [1, 0, 0]]}, {"op": "Activation", "name": "relu0", "attrs": {"act_type": "relu"}, "inputs": [[1, 0, 0]]}, {"op": "Pooling", "name": "pool0", "attrs": {"kernel": "(3, 3)", "pool_type": "max", "stride": "(2, 2)", "pad": "(1, 1)"}, "inputs": [[2, 0, 0]]}, {"op": "FullyConnected", "name": "fc1", "attrs": {"num_hidden": "1000"}, "inputs": [[3, 0, 0], [4, 0, 0], [5, 0, 0]]}, {"op": "SoftmaxOutput", "name": "softmax", "inputs": [[4, 0, 0]]}]}'
          },
          { name: 'model-0000.params', type: 'file', path: '/mxnet-resnet50/model-0000.params' }
        ]
      }
    }
  ],
  paddle: [
    {
      id: 'paddle-ocr',
      label: 'PaddleOCR (Detection)',
      initialFile: '/paddle-ocr/__model__',
      root: {
        name: 'paddle-ocr',
        type: 'directory',
        path: '/paddle-ocr',
        children: [
          {
            name: '__model__',
            type: 'file',
            path: '/paddle-ocr/__model__',
            content: JSON.stringify(
              {
                blocks: [
                  {
                    ops: [
                      {
                        type: 'conv2d',
                        inputs: { Input: ['image'], Filter: ['conv1_weights'] },
                        outputs: { Output: ['conv1_out'] },
                        attrs: {
                          paddings: [1, 1, 1, 1],
                          strides: [1, 1],
                          dilations: [1, 1],
                          groups: 1
                        }
                      },
                      {
                        type: 'relu',
                        inputs: { X: ['conv1_out'] },
                        outputs: { Out: ['relu1_out'] }
                      },
                      {
                        type: 'pool2d',
                        inputs: { X: ['relu1_out'] },
                        outputs: { Out: ['pool1_out'] },
                        attrs: {
                          pooling_type: 'max',
                          ksize: [2, 2],
                          strides: [2, 2],
                          paddings: [0, 0, 0, 0]
                        }
                      },
                      {
                        type: 'elementwise_add',
                        inputs: { X: ['pool1_out'], Y: ['bias1'] },
                        outputs: { Out: ['add1_out'] }
                      },
                      {
                        type: 'batch_norm',
                        inputs: {
                          X: ['add1_out'],
                          Scale: ['bn_scale'],
                          Bias: ['bn_bias'],
                          Mean: ['bn_mean'],
                          Variance: ['bn_var']
                        },
                        outputs: { Y: ['bn_out'] },
                        attrs: { epsilon: 1e-5, momentum: 0.9 }
                      },
                      {
                        type: 'mul',
                        inputs: { X: ['bn_out'], Y: ['fc_weights'] },
                        outputs: { Out: ['mul_out'] },
                        attrs: { x_num_col_dims: 1, y_num_col_dims: 1 }
                      },
                      {
                        type: 'concat',
                        inputs: { X: ['mul_out', 'mul_out'] },
                        outputs: { Out: ['concat_out'] },
                        attrs: { axis: 1 }
                      },
                      {
                        type: 'split',
                        inputs: { X: ['concat_out'] },
                        outputs: { Out: ['split_out1', 'split_out2'] },
                        attrs: { axis: 1, num_or_sections: [1, 1] }
                      },
                      {
                        type: 'matmul',
                        inputs: { X: ['split_out1'], Y: ['split_out2'] },
                        outputs: { Out: ['matmul_out'] },
                        attrs: { transpose_x: false, transpose_y: true }
                      }
                    ]
                  }
                ]
              },
              null,
              2
            )
          },
          { name: 'weight', type: 'file', path: '/paddle-ocr/weight' }
        ]
      }
    }
  ],
  scikitlearn: [
    {
      id: 'sklearn-rf',
      label: 'Random Forest Classifier',
      initialFile: '/sklearn-rf/pipeline.json',
      root: {
        name: 'sklearn-rf',
        type: 'directory',
        path: '/sklearn-rf',
        children: [
          {
            name: 'pipeline.json',
            type: 'file',
            path: '/sklearn-rf/pipeline.json',
            content: `{\n  "model": "RandomForestClassifier",\n  "n_estimators": 100,\n  "max_depth": 5\n}`
          },
          { name: 'train.py', type: 'file', path: '/sklearn-rf/train.py' }
        ]
      }
    },
    {
      id: 'sklearn-svc',
      label: 'SVM Pipeline',
      initialFile: '/sklearn-svc/pipeline.json',
      root: {
        name: 'sklearn-svc',
        type: 'directory',
        path: '/sklearn-svc',
        children: [
          {
            name: 'pipeline.json',
            type: 'file',
            path: '/sklearn-svc/pipeline.json',
            content: `{\n  "model": "SVC",\n  "kernel": "rbf",\n  "C": 1.0\n}`
          }
        ]
      }
    }
  ],
  lightgbm: [
    {
      id: 'lgbm-regressor',
      label: 'LGBM Regressor',
      initialFile: '/lightgbm-reg/model.txt',
      root: {
        name: 'lightgbm-reg',
        type: 'directory',
        path: '/lightgbm-reg',
        children: [
          {
            name: 'model.txt',
            type: 'file',
            path: '/lightgbm-reg/model.txt',
            content: `tree\nversion=v3\nnum_class=1\nnum_tree_per_iteration=1\nlabel_index=0\nmax_feature_idx=4\n`
          },
          { name: 'train.py', type: 'file', path: '/lightgbm-reg/train.py' }
        ]
      }
    }
  ],
  xgboost: [
    {
      id: 'xgb-classifier',
      label: 'XGBoost Classifier',
      initialFile: '/xgb-class/model.json',
      root: {
        name: 'xgb-class',
        type: 'directory',
        path: '/xgb-class',
        children: [
          {
            name: 'model.json',
            type: 'file',
            path: '/xgb-class/model.json',
            content: `{"learner":{"objective":{"name":"binary:logistic"}}}`
          }
        ]
      }
    }
  ],
  catboost: [
    {
      id: 'catboost-model',
      label: 'CatBoost Default',
      initialFile: '/catboost-model/model.json',
      root: {
        name: 'catboost-model',
        type: 'directory',
        path: '/catboost-model',
        children: [
          {
            name: 'model.json',
            type: 'file',
            path: '/catboost-model/model.json',
            content: `{"catboost_version": "1.0.6", "model_info": {"class_names": ["0", "1"]}}`
          }
        ]
      }
    }
  ],
  sparkml: [
    {
      id: 'sparkml-pipeline',
      label: 'SparkML Pipeline',
      initialFile: '/sparkml-pipeline/metadata/part-00000',
      root: {
        name: 'sparkml-pipeline',
        type: 'directory',
        path: '/sparkml-pipeline',
        children: [
          {
            name: 'metadata',
            type: 'directory',
            path: '/sparkml-pipeline/metadata',
            children: [
              {
                name: 'part-00000',
                type: 'file',
                path: '/sparkml-pipeline/metadata/part-00000',
                content: `{"class": "org.apache.spark.ml.classification.LogisticRegressionModel", "numClasses": 2}`
              }
            ]
          },
          { name: 'stages', type: 'directory', path: '/sparkml-pipeline/stages', children: [] }
        ]
      }
    }
  ]
};

export const RHS_TARGETS: Record<string, FileNode> = {
  onnx: {
    name: 'output-onnx',
    type: 'directory',
    path: '/output-onnx',
    children: [
      {
        name: 'model.onnx',
        type: 'file',
        path: '/output-onnx/model.onnx',
        content: `// Binary representation of /output-onnx/model.onnx
// ONNX AST Structure (Mocked output):
// ir_version: 8
// producer_name: "onnx9000-converter"
// graph {
//   node {
//     input: "image_input"
//     output: "conv2d_1_output"
//     op_type: "Conv"
//   }
//   node {
//     input: "conv2d_1_output"
//     output: "max_pooling2d_1_output"
//     op_type: "MaxPool"
//   }
//   ...
//   output {
//     name: "output_probs"
//     type {
//       tensor_type {
//         elem_type: 1
//         shape { dim { dim_value: 1 } dim { dim_value: 10 } }
//       }
//     }
//   }
// }
`
      }
    ]
  },
  olive: {
    name: 'olive-optimized',
    type: 'directory',
    path: '/olive-optimized',
    children: [
      { name: 'optimized_model.onnx', type: 'file', path: '/olive-optimized/optimized_model.onnx' }
    ]
  },
  'onnx-simplifier': {
    name: 'simplified-model',
    type: 'directory',
    path: '/simplified-model',
    children: [{ name: 'simplified.onnx', type: 'file', path: '/simplified-model/simplified.onnx' }]
  },
  mlir: {
    name: 'output-mlir',
    type: 'directory',
    path: '/output-mlir',
    children: [{ name: 'graph.mlir', type: 'file', path: '/output-mlir/graph.mlir' }]
  },

  c: {
    name: 'output-c',
    type: 'directory',
    path: '/output-c',
    children: [
      {
        name: 'model.h',
        type: 'file',
        path: '/output-c/model.h',
        content: `#ifndef MODEL_H\n#define MODEL_H\n\n#ifdef __cplusplus\nextern "C" {\n#endif\n\nvoid run_model(const float* input, float* output);\n\n#ifdef __cplusplus\n}\n#endif\n\n#endif // MODEL_H`
      },
      {
        name: 'model.c',
        type: 'file',
        path: '/output-c/model.c',
        content: `#include "model.h"\n#include <stdio.h>\n#include <stdlib.h>\n\nvoid run_model(const float* input, float* output) {\n    // A simple mock calculation\n    for (int i = 0; i < 10; ++i) {\n        output[i] = input[i] * 2.0f;\n    }\n}\n\nint main() {\n    float input[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};\n    float output[10];\n    \n    run_model(input, output);\n    \n    printf("Output: ");\n    for (int i = 0; i < 10; ++i) {\n        printf("%f ", output[i]);\n    }\n    printf("\\n");\n    \n    return 0;\n}`
      }
    ]
  },
  cpp: {
    name: 'output-cpp',
    type: 'directory',
    path: '/output-cpp',
    children: [
      {
        name: 'model.h',
        type: 'file',
        path: '/output-cpp/model.h',
        content: `#ifndef MODEL_H\n#define MODEL_H\n\n#include <vector>\n\nclass Model {\npublic:\n    void run(const std::vector<float>& input, std::vector<float>& output);\n};\n\n#endif // MODEL_H`
      },
      {
        name: 'model.cpp',
        type: 'file',
        path: '/output-cpp/model.cpp',
        content: `#include "model.h"\n#include <iostream>\n\nvoid Model::run(const std::vector<float>& input, std::vector<float>& output) {\n    output.resize(input.size());\n    for (size_t i = 0; i < input.size(); ++i) {\n        output[i] = input[i] * 3.0f;\n    }\n}\n\nint main() {\n    Model model;\n    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};\n    std::vector<float> output;\n    \n    model.run(input, output);\n    \n    std::cout << "Output: ";\n    for (float val : output) {\n        std::cout << val << " ";\n    }\n    std::cout << "\\n";\n    \n    return 0;\n}`
      }
    ]
  },
  coreml: {
    name: 'model.mlpackage',
    type: 'directory',
    path: '/model.mlpackage',
    children: [{ name: 'Manifest.json', type: 'file', path: '/model.mlpackage/Manifest.json' }]
  },
  caffe: {
    name: 'output-caffe',
    type: 'directory',
    path: '/output-caffe',
    children: [{ name: 'model.prototxt', type: 'file', path: '/output-caffe/model.prototxt' }]
  },
  keras: {
    name: 'output-keras',
    type: 'directory',
    path: '/output-keras',
    children: [{ name: 'model.py', type: 'file', path: '/output-keras/model.py' }]
  },
  mxnet: {
    name: 'output-mxnet',
    type: 'directory',
    path: '/output-mxnet',
    children: [{ name: 'model-symbol.json', type: 'file', path: '/output-mxnet/model-symbol.json' }]
  },
  tensorflow: {
    name: 'output-tf',
    type: 'directory',
    path: '/output-tf',
    children: [{ name: 'saved_model.pb', type: 'file', path: '/output-tf/saved_model.pb' }]
  },
  cntk: {
    name: 'output-cntk',
    type: 'directory',
    path: '/output-cntk',
    children: [{ name: 'model.py', type: 'file', path: '/output-cntk/model.py' }]
  },
  pytorch: {
    name: 'output-pytorch',
    type: 'directory',
    path: '/output-pytorch',
    children: [{ name: 'module.py', type: 'file', path: '/output-pytorch/module.py' }]
  },
  onnxscript: {
    name: 'output-onnxscript',
    type: 'directory',
    path: '/output-onnxscript',
    children: [{ name: 'model.py', type: 'file', path: '/output-onnxscript/model.py' }]
  }
};
