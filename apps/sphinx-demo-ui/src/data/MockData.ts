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
  { value: 'paddlepaddle', label: 'PaddlePaddle' },
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
      initialFile: '/keras-mnist/model.json',
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
        children: [{ name: 'model.py', type: 'file', path: '/spox-mlp/model.py' }]
      }
    }
  ],
  tensorflow: [
    {
      id: 'tf-mobilenet',
      label: 'MobileNet V2',
      initialFile: '/tf-mobilenet/export.py',
      root: {
        name: 'tf-mobilenet',
        type: 'directory',
        path: '/tf-mobilenet',
        children: [
          { name: 'saved_model.pb', type: 'file', path: '/tf-mobilenet/saved_model.pb' },
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
          { name: 'deploy.prototxt', type: 'file', path: '/caffe-alexnet/deploy.prototxt' },
          { name: 'weights.caffemodel', type: 'file', path: '/caffe-alexnet/weights.caffemodel' }
        ]
      }
    }
  ],
  mxnet: [
    {
      id: 'mxnet-resnet',
      label: 'ResNet 152',
      initialFile: '/mxnet-resnet/model-symbol.json',
      root: {
        name: 'mxnet-resnet',
        type: 'directory',
        path: '/mxnet-resnet',
        children: [
          { name: 'model-symbol.json', type: 'file', path: '/mxnet-resnet/model-symbol.json' },
          { name: 'model-0000.params', type: 'file', path: '/mxnet-resnet/model-0000.params' }
        ]
      }
    }
  ],
  paddlepaddle: [
    {
      id: 'paddle-ocr',
      label: 'PaddleOCR (Detection)',
      initialFile: '/paddle-ocr/__model__',
      root: {
        name: 'paddle-ocr',
        type: 'directory',
        path: '/paddle-ocr',
        children: [
          { name: '__model__', type: 'file', path: '/paddle-ocr/__model__' },
          { name: 'weight', type: 'file', path: '/paddle-ocr/weight' }
        ]
      }
    }
  ],
  scikitlearn: [
    {
      id: 'sklearn-rf',
      label: 'Random Forest Classifier',
      initialFile: '/sklearn-rf/train.py',
      root: {
        name: 'sklearn-rf',
        type: 'directory',
        path: '/sklearn-rf',
        children: [
          { name: 'pipeline.pkl', type: 'file', path: '/sklearn-rf/pipeline.pkl' },
          { name: 'train.py', type: 'file', path: '/sklearn-rf/train.py' }
        ]
      }
    },
    {
      id: 'sklearn-svc',
      label: 'SVM Pipeline',
      initialFile: '/sklearn-svc/pipeline.pkl',
      root: {
        name: 'sklearn-svc',
        type: 'directory',
        path: '/sklearn-svc',
        children: [{ name: 'pipeline.pkl', type: 'file', path: '/sklearn-svc/pipeline.pkl' }]
      }
    }
  ],
  lightgbm: [
    {
      id: 'lgbm-regressor',
      label: 'LGBM Regressor',
      initialFile: '/lightgbm-reg/train.py',
      root: {
        name: 'lightgbm-reg',
        type: 'directory',
        path: '/lightgbm-reg',
        children: [
          { name: 'model.txt', type: 'file', path: '/lightgbm-reg/model.txt' },
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
        children: [{ name: 'model.json', type: 'file', path: '/xgb-class/model.json' }]
      }
    }
  ],
  catboost: [
    {
      id: 'catboost-model',
      label: 'CatBoost Default',
      initialFile: '/catboost-model/model.cbm',
      root: {
        name: 'catboost-model',
        type: 'directory',
        path: '/catboost-model',
        children: [{ name: 'model.cbm', type: 'file', path: '/catboost-model/model.cbm' }]
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
              { name: 'part-00000', type: 'file', path: '/sparkml-pipeline/metadata/part-00000' }
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
  cpp: {
    name: 'output-cpp',
    type: 'directory',
    path: '/output-cpp',
    children: [
      { name: 'model.h', type: 'file', path: '/output-cpp/model.h' },
      { name: 'model.cpp', type: 'file', path: '/output-cpp/model.cpp' }
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
    children: [{ name: 'model.h5', type: 'file', path: '/output-keras/model.h5' }]
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
    children: [{ name: 'model.model', type: 'file', path: '/output-cntk/model.model' }]
  },
  pytorch: {
    name: 'output-pytorch',
    type: 'directory',
    path: '/output-pytorch',
    children: [{ name: 'module.py', type: 'file', path: '/output-pytorch/module.py' }]
  }
};
