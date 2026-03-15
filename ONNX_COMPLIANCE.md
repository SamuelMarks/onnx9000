# Exhaustive ONNX Compliance Checklist

This document tracks the progress towards 100% compliance with the [Open Neural Network Exchange (ONNX) specification](https://github.com/onnx/onnx).

**🔥 STATUS:** `onnx9000` is maturing rapidly. Over 200 standard ONNX operators have been implemented. The core IR parsing, autograd engine, static C++ memory planning, Apple Accelerate framework, WebAssembly SIMD backends, CUDA target, WebGPU, and advanced WebWorker RPC architectures are fully integrated and verified with 100% test and doc coverage across Python, C++, and TypeScript.


## 1. Core Protobuf & Serialization
### 1.1 ModelProto
- [x] [x] [x] Parse `ModelProto`
- [x] [x] [x] Serialize `ModelProto`
- [x] [x] [x] Validate `ModelProto` constraints
### 1.2 GraphProto
- [x] [x] [x] Parse `GraphProto`
- [x] [x] [x] Serialize `GraphProto`
- [x] [x] [x] Validate `GraphProto` constraints
### 1.3 NodeProto
- [x] [x] [x] Parse `NodeProto`
- [x] [x] [x] Serialize `NodeProto`
- [x] [x] [x] Validate `NodeProto` constraints
### 1.4 TensorProto
- [x] [x] [x] Parse `TensorProto`
- [x] [x] [x] Serialize `TensorProto`
- [x] [x] [x] Validate `TensorProto` constraints
### 1.5 ValueInfoProto
- [x] [x] [x] Parse `ValueInfoProto`
- [x] [x] [x] Serialize `ValueInfoProto`
- [x] [x] [x] Validate `ValueInfoProto` constraints
### 1.6 AttributeProto
- [x] [x] [x] Parse `AttributeProto`
- [x] [x] [x] Serialize `AttributeProto`
- [x] [x] [x] Validate `AttributeProto` constraints
### 1.7 TensorShapeProto
- [x] [x] [x] Parse `TensorShapeProto`
- [x] [x] [x] Serialize `TensorShapeProto`
- [x] [x] [x] Validate `TensorShapeProto` constraints
### 1.8 TypeProto
- [x] [x] [x] Parse `TypeProto`
- [x] [x] [x] Serialize `TypeProto`
- [x] [x] [x] Validate `TypeProto` constraints
### 1.9 SparseTensorProto
- [x] [x] [x] Parse `SparseTensorProto`
- [x] [x] [x] Serialize `SparseTensorProto`
- [x] [x] [x] Validate `SparseTensorProto` constraints
### 1.10 SequenceProto
- [x] [x] [x] Parse `SequenceProto`
- [x] [x] [x] Serialize `SequenceProto`
- [x] [x] [x] Validate `SequenceProto` constraints
### 1.11 MapProto
- [x] [x] [x] Parse `MapProto`
- [x] [x] [x] Serialize `MapProto`
- [x] [x] [x] Validate `MapProto` constraints
### 1.12 TrainingInfoProto
- [x] [x] [x] Parse `TrainingInfoProto`
- [x] [x] [x] Serialize `TrainingInfoProto`
- [x] [x] [x] Validate `TrainingInfoProto` constraints
### 1.13 FunctionProto
- [x] [x] [x] Parse `FunctionProto`
- [x] [x] [x] Serialize `FunctionProto`
- [x] [x] [x] Validate `FunctionProto` constraints

## 2. Operators


### Domain: `ai.onnx`
#### `Abs`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `Acos`
##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `Acosh`
##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `Add`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `broadcast` (type: AttrType.INT)
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `broadcast` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

##### Opset 14
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `AffineGrid`
##### Opset 20
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `align_corners` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `theta`
  - [x] [x] [x] `size`
- **Outputs:**
  - [x] [x] [x] `grid`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T2` supports: tensor(int64)

#### `And`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `broadcast` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bool)
  - [x] [x] [x] `T1` supports: tensor(bool)

##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bool)
  - [x] [x] [x] `T1` supports: tensor(bool)

#### `ArgMax`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 12
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
  - [x] [x] [x] `select_last_index` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
  - [x] [x] [x] `select_last_index` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `ArgMin`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 12
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
  - [x] [x] [x] `select_last_index` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
  - [x] [x] [x] `select_last_index` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `Asin`
##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `Asinh`
##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `Atan`
##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `Atanh`
##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `Attention`
##### Opset 23
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `is_causal` (type: AttrType.INT)
  - [x] [x] [x] `kv_num_heads` (type: AttrType.INT)
  - [x] [x] [x] `q_num_heads` (type: AttrType.INT)
  - [x] [x] [x] `qk_matmul_output_mode` (type: AttrType.INT)
  - [x] [x] [x] `scale` (type: AttrType.FLOAT)
  - [x] [x] [x] `softcap` (type: AttrType.FLOAT)
  - [x] [x] [x] `softmax_precision` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `Q`
  - [x] [x] [x] `K`
  - [x] [x] [x] `V`
  - [x] [x] [x] `attn_mask`
  - [x] [x] [x] `past_key`
  - [x] [x] [x] `past_value`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `present_key`
  - [x] [x] [x] `present_value`
  - [x] [x] [x] `qk_matmul_output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T2` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `U` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(bool)

##### Opset 24
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `is_causal` (type: AttrType.INT)
  - [x] [x] [x] `kv_num_heads` (type: AttrType.INT)
  - [x] [x] [x] `q_num_heads` (type: AttrType.INT)
  - [x] [x] [x] `qk_matmul_output_mode` (type: AttrType.INT)
  - [x] [x] [x] `scale` (type: AttrType.FLOAT)
  - [x] [x] [x] `softcap` (type: AttrType.FLOAT)
  - [x] [x] [x] `softmax_precision` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `Q`
  - [x] [x] [x] `K`
  - [x] [x] [x] `V`
  - [x] [x] [x] `attn_mask`
  - [x] [x] [x] `past_key`
  - [x] [x] [x] `past_value`
  - [x] [x] [x] `nonpad_kv_seqlen`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `present_key`
  - [x] [x] [x] `present_value`
  - [x] [x] [x] `qk_matmul_output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T2` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `U` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(bool)

#### `AveragePool`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `auto_pad` (type: AttrType.STRING)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `auto_pad` (type: AttrType.STRING)
  - [x] [x] [x] `count_include_pad` (type: AttrType.INT)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 10
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `auto_pad` (type: AttrType.STRING)
  - [x] [x] [x] `ceil_mode` (type: AttrType.INT)
  - [x] [x] [x] `count_include_pad` (type: AttrType.INT)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `auto_pad` (type: AttrType.STRING)
  - [x] [x] [x] `ceil_mode` (type: AttrType.INT)
  - [x] [x] [x] `count_include_pad` (type: AttrType.INT)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 19
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `auto_pad` (type: AttrType.STRING)
  - [x] [x] [x] `ceil_mode` (type: AttrType.INT)
  - [x] [x] [x] `count_include_pad` (type: AttrType.INT)
  - [x] [x] [x] `dilations` (type: AttrType.INTS)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `auto_pad` (type: AttrType.STRING)
  - [x] [x] [x] `ceil_mode` (type: AttrType.INT)
  - [x] [x] [x] `count_include_pad` (type: AttrType.INT)
  - [x] [x] [x] `dilations` (type: AttrType.INTS)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `BatchNormalization`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
  - [x] [x] [x] `epsilon` (type: AttrType.FLOAT)
  - [x] [x] [x] `is_test` (type: AttrType.INT)
  - [x] [x] [x] `momentum` (type: AttrType.FLOAT)
  - [x] [x] [x] `spatial` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `scale`
  - [x] [x] [x] `B`
  - [x] [x] [x] `mean`
  - [x] [x] [x] `var`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `mean`
  - [x] [x] [x] `var`
  - [x] [x] [x] `saved_mean`
  - [x] [x] [x] `saved_var`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `epsilon` (type: AttrType.FLOAT)
  - [x] [x] [x] `is_test` (type: AttrType.INT)
  - [x] [x] [x] `momentum` (type: AttrType.FLOAT)
  - [x] [x] [x] `spatial` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `scale`
  - [x] [x] [x] `B`
  - [x] [x] [x] `mean`
  - [x] [x] [x] `var`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `mean`
  - [x] [x] [x] `var`
  - [x] [x] [x] `saved_mean`
  - [x] [x] [x] `saved_var`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `epsilon` (type: AttrType.FLOAT)
  - [x] [x] [x] `momentum` (type: AttrType.FLOAT)
  - [x] [x] [x] `spatial` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `scale`
  - [x] [x] [x] `B`
  - [x] [x] [x] `mean`
  - [x] [x] [x] `var`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `mean`
  - [x] [x] [x] `var`
  - [x] [x] [x] `saved_mean`
  - [x] [x] [x] `saved_var`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `epsilon` (type: AttrType.FLOAT)
  - [x] [x] [x] `momentum` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `scale`
  - [x] [x] [x] `B`
  - [x] [x] [x] `mean`
  - [x] [x] [x] `var`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `mean`
  - [x] [x] [x] `var`
  - [x] [x] [x] `saved_mean`
  - [x] [x] [x] `saved_var`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 14
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `epsilon` (type: AttrType.FLOAT)
  - [x] [x] [x] `momentum` (type: AttrType.FLOAT)
  - [x] [x] [x] `training_mode` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `scale`
  - [x] [x] [x] `B`
  - [x] [x] [x] `input_mean`
  - [x] [x] [x] `input_var`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `running_mean`
  - [x] [x] [x] `running_var`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
  - [x] [x] [x] `U` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

##### Opset 15
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `epsilon` (type: AttrType.FLOAT)
  - [x] [x] [x] `momentum` (type: AttrType.FLOAT)
  - [x] [x] [x] `training_mode` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `scale`
  - [x] [x] [x] `B`
  - [x] [x] [x] `input_mean`
  - [x] [x] [x] `input_var`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `running_mean`
  - [x] [x] [x] `running_var`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
  - [x] [x] [x] `T1` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
  - [x] [x] [x] `T2` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `Bernoulli`
##### Opset 15
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `dtype` (type: AttrType.INT)
  - [x] [x] [x] `seed` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T2` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bool)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `dtype` (type: AttrType.INT)
  - [x] [x] [x] `seed` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T2` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(bool)

#### `BitShift`
##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `direction` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `Y`
- **Outputs:**
  - [x] [x] [x] `Z`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64)

#### `BitwiseAnd`
##### Opset 18
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64)

#### `BitwiseNot`
##### Opset 18
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64)

#### `BitwiseOr`
##### Opset 18
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64)

#### `BitwiseXor`
##### Opset 18
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64)

#### `BlackmanWindow`
##### Opset 17
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `output_datatype` (type: AttrType.INT)
  - [x] [x] [x] `periodic` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `size`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(int32), tensor(int64)
  - [x] [x] [x] `T2` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `Cast`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `to` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(bool)
  - [x] [x] [x] `T2` supports: tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(bool)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `to` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(bool)
  - [x] [x] [x] `T2` supports: tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(bool)

##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `to` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(bool), tensor(string)
  - [x] [x] [x] `T2` supports: tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(bool), tensor(string)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `to` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(bool), tensor(string), tensor(bfloat16)
  - [x] [x] [x] `T2` supports: tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(bool), tensor(string), tensor(bfloat16)

##### Opset 19
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `saturate` (type: AttrType.INT)
  - [x] [x] [x] `to` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(bool), tensor(string), tensor(bfloat16), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)
  - [x] [x] [x] `T2` supports: tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(bool), tensor(string), tensor(bfloat16), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)

##### Opset 21
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `saturate` (type: AttrType.INT)
  - [x] [x] [x] `to` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(bool), tensor(string), tensor(bfloat16), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4)
  - [x] [x] [x] `T2` supports: tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(bool), tensor(string), tensor(bfloat16), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4)

##### Opset 23
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `saturate` (type: AttrType.INT)
  - [x] [x] [x] `to` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1)
  - [x] [x] [x] `T2` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1)

##### Opset 24
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `round_mode` (type: AttrType.STRING)
  - [x] [x] [x] `saturate` (type: AttrType.INT)
  - [x] [x] [x] `to` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1), tensor(float8e8m0)
  - [x] [x] [x] `T2` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1), tensor(float8e8m0)

#### `CastLike`
##### Opset 15
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `target_type`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(bool), tensor(string), tensor(bfloat16)
  - [x] [x] [x] `T2` supports: tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(bool), tensor(string), tensor(bfloat16)

##### Opset 19
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `saturate` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `target_type`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(bool), tensor(string), tensor(bfloat16), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)
  - [x] [x] [x] `T2` supports: tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(bool), tensor(string), tensor(bfloat16), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)

##### Opset 21
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `saturate` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `target_type`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4)
  - [x] [x] [x] `T2` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4)

##### Opset 23
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `saturate` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `target_type`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1)
  - [x] [x] [x] `T2` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1)

##### Opset 24
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `round_mode` (type: AttrType.STRING)
  - [x] [x] [x] `saturate` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `target_type`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1), tensor(float8e8m0)
  - [x] [x] [x] `T2` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1), tensor(float8e8m0)

#### `Ceil`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `Celu`
##### Opset 12
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `alpha` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float)

#### `CenterCropPad`
##### Opset 18
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `input_data`
  - [x] [x] [x] `shape`
- **Outputs:**
  - [x] [x] [x] `output_data`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `Tind` supports: tensor(int32), tensor(int64)

#### `Clip`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
  - [x] [x] [x] `max` (type: AttrType.FLOAT)
  - [x] [x] [x] `min` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `max` (type: AttrType.FLOAT)
  - [x] [x] [x] `min` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `min`
  - [x] [x] [x] `max`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 12
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `min`
  - [x] [x] [x] `max`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `min`
  - [x] [x] [x] `max`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `Col2Im`
##### Opset 18
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `dilations` (type: AttrType.INTS)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `image_shape`
  - [x] [x] [x] `block_shape`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

#### `Compress`
##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `condition`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `T1` supports: tensor(bool)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `condition`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `T1` supports: tensor(bool)

#### `Concat`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `inputs`
- **Outputs:**
  - [x] [x] [x] `concat_result`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 4
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `inputs`
- **Outputs:**
  - [x] [x] [x] `concat_result`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `inputs`
- **Outputs:**
  - [x] [x] [x] `concat_result`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `inputs`
- **Outputs:**
  - [x] [x] [x] `concat_result`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

#### `ConcatFromSequence`
##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `new_axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input_sequence`
- **Outputs:**
  - [x] [x] [x] `concat_result`
- **Type Constraints:**
  - [x] [x] [x] `S` supports: seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128))
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

#### `Constant`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `value` (type: AttrType.TENSOR)
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `value` (type: AttrType.TENSOR)
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `sparse_value` (type: AttrType.SPARSE_TENSOR)
  - [x] [x] [x] `value` (type: AttrType.TENSOR)
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 12
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `sparse_value` (type: AttrType.SPARSE_TENSOR)
  - [x] [x] [x] `value` (type: AttrType.TENSOR)
  - [x] [x] [x] `value_float` (type: AttrType.FLOAT)
  - [x] [x] [x] `value_floats` (type: AttrType.FLOATS)
  - [x] [x] [x] `value_int` (type: AttrType.INT)
  - [x] [x] [x] `value_ints` (type: AttrType.INTS)
  - [x] [x] [x] `value_string` (type: AttrType.STRING)
  - [x] [x] [x] `value_strings` (type: AttrType.STRINGS)
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `sparse_value` (type: AttrType.SPARSE_TENSOR)
  - [x] [x] [x] `value` (type: AttrType.TENSOR)
  - [x] [x] [x] `value_float` (type: AttrType.FLOAT)
  - [x] [x] [x] `value_floats` (type: AttrType.FLOATS)
  - [x] [x] [x] `value_int` (type: AttrType.INT)
  - [x] [x] [x] `value_ints` (type: AttrType.INTS)
  - [x] [x] [x] `value_string` (type: AttrType.STRING)
  - [x] [x] [x] `value_strings` (type: AttrType.STRINGS)
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 19
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `sparse_value` (type: AttrType.SPARSE_TENSOR)
  - [x] [x] [x] `value` (type: AttrType.TENSOR)
  - [x] [x] [x] `value_float` (type: AttrType.FLOAT)
  - [x] [x] [x] `value_floats` (type: AttrType.FLOATS)
  - [x] [x] [x] `value_int` (type: AttrType.INT)
  - [x] [x] [x] `value_ints` (type: AttrType.INTS)
  - [x] [x] [x] `value_string` (type: AttrType.STRING)
  - [x] [x] [x] `value_strings` (type: AttrType.STRINGS)
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)

##### Opset 21
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `sparse_value` (type: AttrType.SPARSE_TENSOR)
  - [x] [x] [x] `value` (type: AttrType.TENSOR)
  - [x] [x] [x] `value_float` (type: AttrType.FLOAT)
  - [x] [x] [x] `value_floats` (type: AttrType.FLOATS)
  - [x] [x] [x] `value_int` (type: AttrType.INT)
  - [x] [x] [x] `value_ints` (type: AttrType.INTS)
  - [x] [x] [x] `value_string` (type: AttrType.STRING)
  - [x] [x] [x] `value_strings` (type: AttrType.STRINGS)
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4)

##### Opset 23
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `sparse_value` (type: AttrType.SPARSE_TENSOR)
  - [x] [x] [x] `value` (type: AttrType.TENSOR)
  - [x] [x] [x] `value_float` (type: AttrType.FLOAT)
  - [x] [x] [x] `value_floats` (type: AttrType.FLOATS)
  - [x] [x] [x] `value_int` (type: AttrType.INT)
  - [x] [x] [x] `value_ints` (type: AttrType.INTS)
  - [x] [x] [x] `value_string` (type: AttrType.STRING)
  - [x] [x] [x] `value_strings` (type: AttrType.STRINGS)
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1)

##### Opset 24
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `sparse_value` (type: AttrType.SPARSE_TENSOR)
  - [x] [x] [x] `value` (type: AttrType.TENSOR)
  - [x] [x] [x] `value_float` (type: AttrType.FLOAT)
  - [x] [x] [x] `value_floats` (type: AttrType.FLOATS)
  - [x] [x] [x] `value_int` (type: AttrType.INT)
  - [x] [x] [x] `value_ints` (type: AttrType.INTS)
  - [x] [x] [x] `value_string` (type: AttrType.STRING)
  - [x] [x] [x] `value_strings` (type: AttrType.STRINGS)
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1), tensor(float8e8m0)

#### `ConstantOfShape`
##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `value` (type: AttrType.TENSOR)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(int64)
  - [x] [x] [x] `T2` supports: tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(bool)

##### Opset 20
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `value` (type: AttrType.TENSOR)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(int64)
  - [x] [x] [x] `T2` supports: tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(bool), tensor(bfloat16), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)

##### Opset 21
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `value` (type: AttrType.TENSOR)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(int64)
  - [x] [x] [x] `T2` supports: tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint4), tensor(int4), tensor(bool), tensor(bfloat16), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)

##### Opset 23
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `value` (type: AttrType.TENSOR)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(int64)
  - [x] [x] [x] `T2` supports: tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint4), tensor(int4), tensor(bool), tensor(bfloat16), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(float4e2m1)

##### Opset 24
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `value` (type: AttrType.TENSOR)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(int64)
  - [x] [x] [x] `T2` supports: tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint4), tensor(int4), tensor(bool), tensor(bfloat16), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(float4e2m1), tensor(float8e8m0)

#### `Conv`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `auto_pad` (type: AttrType.STRING)
  - [x] [x] [x] `dilations` (type: AttrType.INTS)
  - [x] [x] [x] `group` (type: AttrType.INT)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `W`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `auto_pad` (type: AttrType.STRING)
  - [x] [x] [x] `dilations` (type: AttrType.INTS)
  - [x] [x] [x] `group` (type: AttrType.INT)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `W`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `auto_pad` (type: AttrType.STRING)
  - [x] [x] [x] `dilations` (type: AttrType.INTS)
  - [x] [x] [x] `group` (type: AttrType.INT)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `W`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `ConvInteger`
##### Opset 10
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `auto_pad` (type: AttrType.STRING)
  - [x] [x] [x] `dilations` (type: AttrType.INTS)
  - [x] [x] [x] `group` (type: AttrType.INT)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `x`
  - [x] [x] [x] `w`
  - [x] [x] [x] `x_zero_point`
  - [x] [x] [x] `w_zero_point`
- **Outputs:**
  - [x] [x] [x] `y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(int8), tensor(uint8)
  - [x] [x] [x] `T2` supports: tensor(int8), tensor(uint8)
  - [x] [x] [x] `T3` supports: tensor(int32)

#### `ConvTranspose`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `auto_pad` (type: AttrType.STRING)
  - [x] [x] [x] `dilations` (type: AttrType.INTS)
  - [x] [x] [x] `group` (type: AttrType.INT)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `output_padding` (type: AttrType.INTS)
  - [x] [x] [x] `output_shape` (type: AttrType.INTS)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `W`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `auto_pad` (type: AttrType.STRING)
  - [x] [x] [x] `dilations` (type: AttrType.INTS)
  - [x] [x] [x] `group` (type: AttrType.INT)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `output_padding` (type: AttrType.INTS)
  - [x] [x] [x] `output_shape` (type: AttrType.INTS)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `W`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `auto_pad` (type: AttrType.STRING)
  - [x] [x] [x] `dilations` (type: AttrType.INTS)
  - [x] [x] [x] `group` (type: AttrType.INT)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `output_padding` (type: AttrType.INTS)
  - [x] [x] [x] `output_shape` (type: AttrType.INTS)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `W`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `Cos`
##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `Cosh`
##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `CumSum`
##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `exclusive` (type: AttrType.INT)
  - [x] [x] [x] `reverse` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `x`
  - [x] [x] [x] `axis`
- **Outputs:**
  - [x] [x] [x] `y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float), tensor(double)
  - [x] [x] [x] `T2` supports: tensor(int32), tensor(int64)

##### Opset 14
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `exclusive` (type: AttrType.INT)
  - [x] [x] [x] `reverse` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `x`
  - [x] [x] [x] `axis`
- **Outputs:**
  - [x] [x] [x] `y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
  - [x] [x] [x] `T2` supports: tensor(int32), tensor(int64)

#### `DFT`
##### Opset 17
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `inverse` (type: AttrType.INT)
  - [x] [x] [x] `onesided` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `dft_length`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
  - [x] [x] [x] `T2` supports: tensor(int32), tensor(int64)

##### Opset 20
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `inverse` (type: AttrType.INT)
  - [x] [x] [x] `onesided` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `dft_length`
  - [x] [x] [x] `axis`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T2` supports: tensor(int32), tensor(int64)

#### `DeformConv`
##### Opset 19
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `dilations` (type: AttrType.INTS)
  - [x] [x] [x] `group` (type: AttrType.INT)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `offset_group` (type: AttrType.INT)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `W`
  - [x] [x] [x] `offset`
  - [x] [x] [x] `B`
  - [x] [x] [x] `mask`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `dilations` (type: AttrType.INTS)
  - [x] [x] [x] `group` (type: AttrType.INT)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `offset_group` (type: AttrType.INT)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `W`
  - [x] [x] [x] `offset`
  - [x] [x] [x] `B`
  - [x] [x] [x] `mask`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `DepthToSpace`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `blocksize` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `blocksize` (type: AttrType.INT)
  - [x] [x] [x] `mode` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `blocksize` (type: AttrType.INT)
  - [x] [x] [x] `mode` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

#### `DequantizeLinear`
##### Opset 10
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `x`
  - [x] [x] [x] `x_scale`
  - [x] [x] [x] `x_zero_point`
- **Outputs:**
  - [x] [x] [x] `y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(int8), tensor(uint8), tensor(int32)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `x`
  - [x] [x] [x] `x_scale`
  - [x] [x] [x] `x_zero_point`
- **Outputs:**
  - [x] [x] [x] `y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(int8), tensor(uint8), tensor(int32)

##### Opset 19
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `x`
  - [x] [x] [x] `x_scale`
  - [x] [x] [x] `x_zero_point`
- **Outputs:**
  - [x] [x] [x] `y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(int8), tensor(uint8), tensor(int32), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)
  - [x] [x] [x] `T2` supports: tensor(float), tensor(float16), tensor(bfloat16)

##### Opset 21
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `block_size` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `x`
  - [x] [x] [x] `x_scale`
  - [x] [x] [x] `x_zero_point`
- **Outputs:**
  - [x] [x] [x] `y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(int8), tensor(uint8), tensor(int16), tensor(uint16), tensor(int32), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4)
  - [x] [x] [x] `T2` supports: tensor(float), tensor(float16), tensor(bfloat16)

##### Opset 23
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `block_size` (type: AttrType.INT)
  - [x] [x] [x] `output_dtype` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `x`
  - [x] [x] [x] `x_scale`
  - [x] [x] [x] `x_zero_point`
- **Outputs:**
  - [x] [x] [x] `y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(int8), tensor(uint8), tensor(int16), tensor(uint16), tensor(int32), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1)
  - [x] [x] [x] `T2` supports: tensor(float), tensor(float16), tensor(bfloat16)
  - [x] [x] [x] `T3` supports: tensor(float), tensor(float16), tensor(bfloat16)

##### Opset 24
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `block_size` (type: AttrType.INT)
  - [x] [x] [x] `output_dtype` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `x`
  - [x] [x] [x] `x_scale`
  - [x] [x] [x] `x_zero_point`
- **Outputs:**
  - [x] [x] [x] `y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(int8), tensor(uint8), tensor(int16), tensor(uint16), tensor(int32), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1)
  - [x] [x] [x] `T2` supports: tensor(float), tensor(float16), tensor(bfloat16), tensor(float8e8m0)
  - [x] [x] [x] `T3` supports: tensor(float), tensor(float16), tensor(bfloat16)

#### `Det`
##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `Div`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `broadcast` (type: AttrType.INT)
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `broadcast` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

##### Opset 14
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `Dropout`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
  - [x] [x] [x] `is_test` (type: AttrType.INT)
  - [x] [x] [x] `ratio` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `output`
  - [x] [x] [x] `mask`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `is_test` (type: AttrType.INT)
  - [x] [x] [x] `ratio` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `output`
  - [x] [x] [x] `mask`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `ratio` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `output`
  - [x] [x] [x] `mask`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 10
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `ratio` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `output`
  - [x] [x] [x] `mask`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T1` supports: tensor(bool)

##### Opset 12
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `seed` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `ratio`
  - [x] [x] [x] `training_mode`
- **Outputs:**
  - [x] [x] [x] `output`
  - [x] [x] [x] `mask`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T1` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T2` supports: tensor(bool)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `seed` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `ratio`
  - [x] [x] [x] `training_mode`
- **Outputs:**
  - [x] [x] [x] `output`
  - [x] [x] [x] `mask`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
  - [x] [x] [x] `T1` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T2` supports: tensor(bool)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `seed` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `ratio`
  - [x] [x] [x] `training_mode`
- **Outputs:**
  - [x] [x] [x] `output`
  - [x] [x] [x] `mask`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)
  - [x] [x] [x] `T1` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)
  - [x] [x] [x] `T2` supports: tensor(bool)

#### `DynamicQuantizeLinear`
##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `x`
- **Outputs:**
  - [x] [x] [x] `y`
  - [x] [x] [x] `y_scale`
  - [x] [x] [x] `y_zero_point`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float)
  - [x] [x] [x] `T2` supports: tensor(uint8)

#### `Einsum`
##### Opset 12
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `equation` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `Inputs`
- **Outputs:**
  - [x] [x] [x] `Output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

#### `Elu`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `alpha` (type: AttrType.FLOAT)
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `alpha` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `alpha` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `Equal`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `broadcast` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bool), tensor(int32), tensor(int64)
  - [x] [x] [x] `T1` supports: tensor(bool)

##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bool), tensor(int32), tensor(int64)
  - [x] [x] [x] `T1` supports: tensor(bool)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bool), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T1` supports: tensor(bool)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bool), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
  - [x] [x] [x] `T1` supports: tensor(bool)

##### Opset 19
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bool), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16), tensor(string)
  - [x] [x] [x] `T1` supports: tensor(bool)

#### `Erf`
##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `Exp`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `Expand`
##### Opset 8
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `shape`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `shape`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

#### `EyeLike`
##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `dtype` (type: AttrType.INT)
  - [x] [x] [x] `k` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(bool)
  - [x] [x] [x] `T2` supports: tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(bool)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `dtype` (type: AttrType.INT)
  - [x] [x] [x] `k` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(bool)
  - [x] [x] [x] `T2` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(bool)

#### `Flatten`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 21
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4)

##### Opset 23
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1)

##### Opset 24
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1), tensor(float8e8m0)

#### `Floor`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `GRU`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `activation_alpha` (type: AttrType.FLOATS)
  - [x] [x] [x] `activation_beta` (type: AttrType.FLOATS)
  - [x] [x] [x] `activations` (type: AttrType.STRINGS)
  - [x] [x] [x] `clip` (type: AttrType.FLOAT)
  - [x] [x] [x] `direction` (type: AttrType.STRING)
  - [x] [x] [x] `hidden_size` (type: AttrType.INT)
  - [x] [x] [x] `output_sequence` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `W`
  - [x] [x] [x] `R`
  - [x] [x] [x] `B`
  - [x] [x] [x] `sequence_lens`
  - [x] [x] [x] `initial_h`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `Y_h`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T1` supports: tensor(int32)

##### Opset 3
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `activation_alpha` (type: AttrType.FLOATS)
  - [x] [x] [x] `activation_beta` (type: AttrType.FLOATS)
  - [x] [x] [x] `activations` (type: AttrType.STRINGS)
  - [x] [x] [x] `clip` (type: AttrType.FLOAT)
  - [x] [x] [x] `direction` (type: AttrType.STRING)
  - [x] [x] [x] `hidden_size` (type: AttrType.INT)
  - [x] [x] [x] `linear_before_reset` (type: AttrType.INT)
  - [x] [x] [x] `output_sequence` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `W`
  - [x] [x] [x] `R`
  - [x] [x] [x] `B`
  - [x] [x] [x] `sequence_lens`
  - [x] [x] [x] `initial_h`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `Y_h`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T1` supports: tensor(int32)

##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `activation_alpha` (type: AttrType.FLOATS)
  - [x] [x] [x] `activation_beta` (type: AttrType.FLOATS)
  - [x] [x] [x] `activations` (type: AttrType.STRINGS)
  - [x] [x] [x] `clip` (type: AttrType.FLOAT)
  - [x] [x] [x] `direction` (type: AttrType.STRING)
  - [x] [x] [x] `hidden_size` (type: AttrType.INT)
  - [x] [x] [x] `linear_before_reset` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `W`
  - [x] [x] [x] `R`
  - [x] [x] [x] `B`
  - [x] [x] [x] `sequence_lens`
  - [x] [x] [x] `initial_h`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `Y_h`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T1` supports: tensor(int32)

##### Opset 14
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `activation_alpha` (type: AttrType.FLOATS)
  - [x] [x] [x] `activation_beta` (type: AttrType.FLOATS)
  - [x] [x] [x] `activations` (type: AttrType.STRINGS)
  - [x] [x] [x] `clip` (type: AttrType.FLOAT)
  - [x] [x] [x] `direction` (type: AttrType.STRING)
  - [x] [x] [x] `hidden_size` (type: AttrType.INT)
  - [x] [x] [x] `layout` (type: AttrType.INT)
  - [x] [x] [x] `linear_before_reset` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `W`
  - [x] [x] [x] `R`
  - [x] [x] [x] `B`
  - [x] [x] [x] `sequence_lens`
  - [x] [x] [x] `initial_h`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `Y_h`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T1` supports: tensor(int32)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `activation_alpha` (type: AttrType.FLOATS)
  - [x] [x] [x] `activation_beta` (type: AttrType.FLOATS)
  - [x] [x] [x] `activations` (type: AttrType.STRINGS)
  - [x] [x] [x] `clip` (type: AttrType.FLOAT)
  - [x] [x] [x] `direction` (type: AttrType.STRING)
  - [x] [x] [x] `hidden_size` (type: AttrType.INT)
  - [x] [x] [x] `layout` (type: AttrType.INT)
  - [x] [x] [x] `linear_before_reset` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `W`
  - [x] [x] [x] `R`
  - [x] [x] [x] `B`
  - [x] [x] [x] `sequence_lens`
  - [x] [x] [x] `initial_h`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `Y_h`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T1` supports: tensor(int32)

#### `Gather`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `indices`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `Tind` supports: tensor(int32), tensor(int64)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `indices`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `Tind` supports: tensor(int32), tensor(int64)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `indices`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `Tind` supports: tensor(int32), tensor(int64)

#### `GatherElements`
##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `indices`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `Tind` supports: tensor(int32), tensor(int64)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `indices`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `Tind` supports: tensor(int32), tensor(int64)

#### `GatherND`
##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `indices`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 12
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `batch_dims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `indices`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `batch_dims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `indices`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

#### `Gelu`
##### Opset 20
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `approximate` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `Gemm`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `alpha` (type: AttrType.FLOAT)
  - [x] [x] [x] `beta` (type: AttrType.FLOAT)
  - [x] [x] [x] `broadcast` (type: AttrType.INT)
  - [x] [x] [x] `transA` (type: AttrType.INT)
  - [x] [x] [x] `transB` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
  - [x] [x] [x] `C`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `alpha` (type: AttrType.FLOAT)
  - [x] [x] [x] `beta` (type: AttrType.FLOAT)
  - [x] [x] [x] `broadcast` (type: AttrType.INT)
  - [x] [x] [x] `transA` (type: AttrType.INT)
  - [x] [x] [x] `transB` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
  - [x] [x] [x] `C`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `alpha` (type: AttrType.FLOAT)
  - [x] [x] [x] `beta` (type: AttrType.FLOAT)
  - [x] [x] [x] `transA` (type: AttrType.INT)
  - [x] [x] [x] `transB` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
  - [x] [x] [x] `C`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `alpha` (type: AttrType.FLOAT)
  - [x] [x] [x] `beta` (type: AttrType.FLOAT)
  - [x] [x] [x] `transA` (type: AttrType.INT)
  - [x] [x] [x] `transB` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
  - [x] [x] [x] `C`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(uint32), tensor(uint64), tensor(int32), tensor(int64)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `alpha` (type: AttrType.FLOAT)
  - [x] [x] [x] `beta` (type: AttrType.FLOAT)
  - [x] [x] [x] `transA` (type: AttrType.INT)
  - [x] [x] [x] `transB` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
  - [x] [x] [x] `C`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(uint32), tensor(uint64), tensor(int32), tensor(int64)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `alpha` (type: AttrType.FLOAT)
  - [x] [x] [x] `beta` (type: AttrType.FLOAT)
  - [x] [x] [x] `transA` (type: AttrType.INT)
  - [x] [x] [x] `transB` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
  - [x] [x] [x] `C`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(bfloat16)

#### `GlobalAveragePool`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `GlobalLpPool`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `p` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 2
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `p` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `p` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `GlobalMaxPool`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `Greater`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `broadcast` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T1` supports: tensor(bool)

##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T1` supports: tensor(bool)

##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T1` supports: tensor(bool)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
  - [x] [x] [x] `T1` supports: tensor(bool)

#### `GreaterOrEqual`
##### Opset 12
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T1` supports: tensor(bool)

##### Opset 16
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
  - [x] [x] [x] `T1` supports: tensor(bool)

#### `GridSample`
##### Opset 16
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `align_corners` (type: AttrType.INT)
  - [x] [x] [x] `mode` (type: AttrType.STRING)
  - [x] [x] [x] `padding_mode` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `grid`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `T2` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 20
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `align_corners` (type: AttrType.INT)
  - [x] [x] [x] `mode` (type: AttrType.STRING)
  - [x] [x] [x] `padding_mode` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `grid`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `T2` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `align_corners` (type: AttrType.INT)
  - [x] [x] [x] `mode` (type: AttrType.STRING)
  - [x] [x] [x] `padding_mode` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `grid`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `T2` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `GroupNormalization`
##### Opset 18
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `epsilon` (type: AttrType.FLOAT)
  - [x] [x] [x] `num_groups` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `scale`
  - [x] [x] [x] `bias`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

##### Opset 21
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `epsilon` (type: AttrType.FLOAT)
  - [x] [x] [x] `num_groups` (type: AttrType.INT)
  - [x] [x] [x] `stash_type` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `scale`
  - [x] [x] [x] `bias`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `HammingWindow`
##### Opset 17
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `output_datatype` (type: AttrType.INT)
  - [x] [x] [x] `periodic` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `size`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(int32), tensor(int64)
  - [x] [x] [x] `T2` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `HannWindow`
##### Opset 17
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `output_datatype` (type: AttrType.INT)
  - [x] [x] [x] `periodic` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `size`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(int32), tensor(int64)
  - [x] [x] [x] `T2` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `HardSigmoid`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `alpha` (type: AttrType.FLOAT)
  - [x] [x] [x] `beta` (type: AttrType.FLOAT)
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `alpha` (type: AttrType.FLOAT)
  - [x] [x] [x] `beta` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `alpha` (type: AttrType.FLOAT)
  - [x] [x] [x] `beta` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `HardSwish`
##### Opset 14
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `Hardmax`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `Identity`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 14
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128))

##### Opset 16
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128)), optional(seq(tensor(uint8))), optional(seq(tensor(uint16))), optional(seq(tensor(uint32))), optional(seq(tensor(uint64))), optional(seq(tensor(int8))), optional(seq(tensor(int16))), optional(seq(tensor(int32))), optional(seq(tensor(int64))), optional(seq(tensor(float16))), optional(seq(tensor(float))), optional(seq(tensor(double))), optional(seq(tensor(string))), optional(seq(tensor(bool))), optional(seq(tensor(complex64))), optional(seq(tensor(complex128))), optional(tensor(uint8)), optional(tensor(uint16)), optional(tensor(uint32)), optional(tensor(uint64)), optional(tensor(int8)), optional(tensor(int16)), optional(tensor(int32)), optional(tensor(int64)), optional(tensor(float16)), optional(tensor(float)), optional(tensor(double)), optional(tensor(string)), optional(tensor(bool)), optional(tensor(complex64)), optional(tensor(complex128))

##### Opset 19
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128)), optional(seq(tensor(uint8))), optional(seq(tensor(uint16))), optional(seq(tensor(uint32))), optional(seq(tensor(uint64))), optional(seq(tensor(int8))), optional(seq(tensor(int16))), optional(seq(tensor(int32))), optional(seq(tensor(int64))), optional(seq(tensor(float16))), optional(seq(tensor(float))), optional(seq(tensor(double))), optional(seq(tensor(string))), optional(seq(tensor(bool))), optional(seq(tensor(complex64))), optional(seq(tensor(complex128))), optional(tensor(uint8)), optional(tensor(uint16)), optional(tensor(uint32)), optional(tensor(uint64)), optional(tensor(int8)), optional(tensor(int16)), optional(tensor(int32)), optional(tensor(int64)), optional(tensor(float16)), optional(tensor(float)), optional(tensor(double)), optional(tensor(string)), optional(tensor(bool)), optional(tensor(complex64)), optional(tensor(complex128))

##### Opset 21
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128)), optional(seq(tensor(uint8))), optional(seq(tensor(uint16))), optional(seq(tensor(uint32))), optional(seq(tensor(uint64))), optional(seq(tensor(int8))), optional(seq(tensor(int16))), optional(seq(tensor(int32))), optional(seq(tensor(int64))), optional(seq(tensor(float16))), optional(seq(tensor(float))), optional(seq(tensor(double))), optional(seq(tensor(string))), optional(seq(tensor(bool))), optional(seq(tensor(complex64))), optional(seq(tensor(complex128))), optional(tensor(uint8)), optional(tensor(uint16)), optional(tensor(uint32)), optional(tensor(uint64)), optional(tensor(int8)), optional(tensor(int16)), optional(tensor(int32)), optional(tensor(int64)), optional(tensor(float16)), optional(tensor(float)), optional(tensor(double)), optional(tensor(string)), optional(tensor(bool)), optional(tensor(complex64)), optional(tensor(complex128))

##### Opset 23
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1), seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128)), optional(seq(tensor(uint8))), optional(seq(tensor(uint16))), optional(seq(tensor(uint32))), optional(seq(tensor(uint64))), optional(seq(tensor(int8))), optional(seq(tensor(int16))), optional(seq(tensor(int32))), optional(seq(tensor(int64))), optional(seq(tensor(float16))), optional(seq(tensor(float))), optional(seq(tensor(double))), optional(seq(tensor(string))), optional(seq(tensor(bool))), optional(seq(tensor(complex64))), optional(seq(tensor(complex128))), optional(tensor(uint8)), optional(tensor(uint16)), optional(tensor(uint32)), optional(tensor(uint64)), optional(tensor(int8)), optional(tensor(int16)), optional(tensor(int32)), optional(tensor(int64)), optional(tensor(float16)), optional(tensor(float)), optional(tensor(double)), optional(tensor(string)), optional(tensor(bool)), optional(tensor(complex64)), optional(tensor(complex128))

##### Opset 24
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1), tensor(float8e8m0), seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128)), optional(seq(tensor(uint8))), optional(seq(tensor(uint16))), optional(seq(tensor(uint32))), optional(seq(tensor(uint64))), optional(seq(tensor(int8))), optional(seq(tensor(int16))), optional(seq(tensor(int32))), optional(seq(tensor(int64))), optional(seq(tensor(float16))), optional(seq(tensor(float))), optional(seq(tensor(double))), optional(seq(tensor(string))), optional(seq(tensor(bool))), optional(seq(tensor(complex64))), optional(seq(tensor(complex128))), optional(tensor(uint8)), optional(tensor(uint16)), optional(tensor(uint32)), optional(tensor(uint64)), optional(tensor(int8)), optional(tensor(int16)), optional(tensor(int32)), optional(tensor(int64)), optional(tensor(float16)), optional(tensor(float)), optional(tensor(double)), optional(tensor(string)), optional(tensor(bool)), optional(tensor(complex64)), optional(tensor(complex128))

#### `If`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `else_branch` (type: AttrType.GRAPH)
  - [x] [x] [x] `then_branch` (type: AttrType.GRAPH)
- **Inputs:**
  - [x] [x] [x] `cond`
- **Outputs:**
  - [x] [x] [x] `outputs`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `B` supports: tensor(bool)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `else_branch` (type: AttrType.GRAPH)
  - [x] [x] [x] `then_branch` (type: AttrType.GRAPH)
- **Inputs:**
  - [x] [x] [x] `cond`
- **Outputs:**
  - [x] [x] [x] `outputs`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `B` supports: tensor(bool)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `else_branch` (type: AttrType.GRAPH)
  - [x] [x] [x] `then_branch` (type: AttrType.GRAPH)
- **Inputs:**
  - [x] [x] [x] `cond`
- **Outputs:**
  - [x] [x] [x] `outputs`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128))
  - [x] [x] [x] `B` supports: tensor(bool)

##### Opset 16
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `else_branch` (type: AttrType.GRAPH)
  - [x] [x] [x] `then_branch` (type: AttrType.GRAPH)
- **Inputs:**
  - [x] [x] [x] `cond`
- **Outputs:**
  - [x] [x] [x] `outputs`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(bfloat16)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128)), optional(seq(tensor(uint8))), optional(seq(tensor(uint16))), optional(seq(tensor(uint32))), optional(seq(tensor(uint64))), optional(seq(tensor(int8))), optional(seq(tensor(int16))), optional(seq(tensor(int32))), optional(seq(tensor(int64))), optional(seq(tensor(bfloat16))), optional(seq(tensor(float16))), optional(seq(tensor(float))), optional(seq(tensor(double))), optional(seq(tensor(string))), optional(seq(tensor(bool))), optional(seq(tensor(complex64))), optional(seq(tensor(complex128))), optional(tensor(uint8)), optional(tensor(uint16)), optional(tensor(uint32)), optional(tensor(uint64)), optional(tensor(int8)), optional(tensor(int16)), optional(tensor(int32)), optional(tensor(int64)), optional(tensor(bfloat16)), optional(tensor(float16)), optional(tensor(float)), optional(tensor(double)), optional(tensor(string)), optional(tensor(bool)), optional(tensor(complex64)), optional(tensor(complex128))
  - [x] [x] [x] `B` supports: tensor(bool)

##### Opset 19
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `else_branch` (type: AttrType.GRAPH)
  - [x] [x] [x] `then_branch` (type: AttrType.GRAPH)
- **Inputs:**
  - [x] [x] [x] `cond`
- **Outputs:**
  - [x] [x] [x] `outputs`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(bfloat16)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128)), seq(tensor(float8e4m3fn)), seq(tensor(float8e4m3fnuz)), seq(tensor(float8e5m2)), seq(tensor(float8e5m2fnuz)), optional(seq(tensor(uint8))), optional(seq(tensor(uint16))), optional(seq(tensor(uint32))), optional(seq(tensor(uint64))), optional(seq(tensor(int8))), optional(seq(tensor(int16))), optional(seq(tensor(int32))), optional(seq(tensor(int64))), optional(seq(tensor(bfloat16))), optional(seq(tensor(float16))), optional(seq(tensor(float))), optional(seq(tensor(double))), optional(seq(tensor(string))), optional(seq(tensor(bool))), optional(seq(tensor(complex64))), optional(seq(tensor(complex128))), optional(tensor(uint8)), optional(tensor(uint16)), optional(tensor(uint32)), optional(tensor(uint64)), optional(tensor(int8)), optional(tensor(int16)), optional(tensor(int32)), optional(tensor(int64)), optional(tensor(bfloat16)), optional(tensor(float16)), optional(tensor(float)), optional(tensor(double)), optional(tensor(string)), optional(tensor(bool)), optional(tensor(complex64)), optional(tensor(complex128)), optional(tensor(float8e4m3fn)), optional(tensor(float8e4m3fnuz)), optional(tensor(float8e5m2)), optional(tensor(float8e5m2fnuz))
  - [x] [x] [x] `B` supports: tensor(bool)

##### Opset 21
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `else_branch` (type: AttrType.GRAPH)
  - [x] [x] [x] `then_branch` (type: AttrType.GRAPH)
- **Inputs:**
  - [x] [x] [x] `cond`
- **Outputs:**
  - [x] [x] [x] `outputs`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(bfloat16)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128)), seq(tensor(float8e4m3fn)), seq(tensor(float8e4m3fnuz)), seq(tensor(float8e5m2)), seq(tensor(float8e5m2fnuz)), seq(tensor(uint4)), seq(tensor(int4)), optional(seq(tensor(uint8))), optional(seq(tensor(uint16))), optional(seq(tensor(uint32))), optional(seq(tensor(uint64))), optional(seq(tensor(int8))), optional(seq(tensor(int16))), optional(seq(tensor(int32))), optional(seq(tensor(int64))), optional(seq(tensor(bfloat16))), optional(seq(tensor(float16))), optional(seq(tensor(float))), optional(seq(tensor(double))), optional(seq(tensor(string))), optional(seq(tensor(bool))), optional(seq(tensor(complex64))), optional(seq(tensor(complex128))), optional(tensor(uint8)), optional(tensor(uint16)), optional(tensor(uint32)), optional(tensor(uint64)), optional(tensor(int8)), optional(tensor(int16)), optional(tensor(int32)), optional(tensor(int64)), optional(tensor(bfloat16)), optional(tensor(float16)), optional(tensor(float)), optional(tensor(double)), optional(tensor(string)), optional(tensor(bool)), optional(tensor(complex64)), optional(tensor(complex128)), optional(tensor(float8e4m3fn)), optional(tensor(float8e4m3fnuz)), optional(tensor(float8e5m2)), optional(tensor(float8e5m2fnuz)), optional(tensor(uint4)), optional(tensor(int4))
  - [x] [x] [x] `B` supports: tensor(bool)

##### Opset 23
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `else_branch` (type: AttrType.GRAPH)
  - [x] [x] [x] `then_branch` (type: AttrType.GRAPH)
- **Inputs:**
  - [x] [x] [x] `cond`
- **Outputs:**
  - [x] [x] [x] `outputs`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1), seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(bfloat16)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128)), seq(tensor(float8e4m3fn)), seq(tensor(float8e4m3fnuz)), seq(tensor(float8e5m2)), seq(tensor(float8e5m2fnuz)), seq(tensor(uint4)), seq(tensor(int4)), seq(tensor(float4e2m1)), optional(seq(tensor(uint8))), optional(seq(tensor(uint16))), optional(seq(tensor(uint32))), optional(seq(tensor(uint64))), optional(seq(tensor(int8))), optional(seq(tensor(int16))), optional(seq(tensor(int32))), optional(seq(tensor(int64))), optional(seq(tensor(bfloat16))), optional(seq(tensor(float16))), optional(seq(tensor(float))), optional(seq(tensor(double))), optional(seq(tensor(string))), optional(seq(tensor(bool))), optional(seq(tensor(complex64))), optional(seq(tensor(complex128))), optional(tensor(uint8)), optional(tensor(uint16)), optional(tensor(uint32)), optional(tensor(uint64)), optional(tensor(int8)), optional(tensor(int16)), optional(tensor(int32)), optional(tensor(int64)), optional(tensor(bfloat16)), optional(tensor(float16)), optional(tensor(float)), optional(tensor(double)), optional(tensor(string)), optional(tensor(bool)), optional(tensor(complex64)), optional(tensor(complex128)), optional(tensor(float8e4m3fn)), optional(tensor(float8e4m3fnuz)), optional(tensor(float8e5m2)), optional(tensor(float8e5m2fnuz)), optional(tensor(uint4)), optional(tensor(int4)), optional(tensor(float4e2m1))
  - [x] [x] [x] `B` supports: tensor(bool)

##### Opset 24
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `else_branch` (type: AttrType.GRAPH)
  - [x] [x] [x] `then_branch` (type: AttrType.GRAPH)
- **Inputs:**
  - [x] [x] [x] `cond`
- **Outputs:**
  - [x] [x] [x] `outputs`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1), tensor(float8e8m0), seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(bfloat16)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128)), seq(tensor(float8e4m3fn)), seq(tensor(float8e4m3fnuz)), seq(tensor(float8e5m2)), seq(tensor(float8e5m2fnuz)), seq(tensor(uint4)), seq(tensor(int4)), seq(tensor(float4e2m1)), seq(tensor(float8e8m0)), optional(seq(tensor(uint8))), optional(seq(tensor(uint16))), optional(seq(tensor(uint32))), optional(seq(tensor(uint64))), optional(seq(tensor(int8))), optional(seq(tensor(int16))), optional(seq(tensor(int32))), optional(seq(tensor(int64))), optional(seq(tensor(bfloat16))), optional(seq(tensor(float16))), optional(seq(tensor(float))), optional(seq(tensor(double))), optional(seq(tensor(string))), optional(seq(tensor(bool))), optional(seq(tensor(complex64))), optional(seq(tensor(complex128))), optional(tensor(uint8)), optional(tensor(uint16)), optional(tensor(uint32)), optional(tensor(uint64)), optional(tensor(int8)), optional(tensor(int16)), optional(tensor(int32)), optional(tensor(int64)), optional(tensor(bfloat16)), optional(tensor(float16)), optional(tensor(float)), optional(tensor(double)), optional(tensor(string)), optional(tensor(bool)), optional(tensor(complex64)), optional(tensor(complex128)), optional(tensor(float8e4m3fn)), optional(tensor(float8e4m3fnuz)), optional(tensor(float8e5m2)), optional(tensor(float8e5m2fnuz)), optional(tensor(uint4)), optional(tensor(int4)), optional(tensor(float4e2m1)), optional(tensor(float8e8m0))
  - [x] [x] [x] `B` supports: tensor(bool)

#### `ImageDecoder`
##### Opset 20
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `pixel_format` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `encoded_stream`
- **Outputs:**
  - [x] [x] [x] `image`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(uint8)
  - [x] [x] [x] `T2` supports: tensor(uint8)

#### `InstanceNormalization`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
  - [x] [x] [x] `epsilon` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `scale`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `epsilon` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `scale`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `epsilon` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `scale`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `IsInf`
##### Opset 10
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `detect_negative` (type: AttrType.INT)
  - [x] [x] [x] `detect_positive` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float), tensor(double)
  - [x] [x] [x] `T2` supports: tensor(bool)

##### Opset 20
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `detect_negative` (type: AttrType.INT)
  - [x] [x] [x] `detect_positive` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)
  - [x] [x] [x] `T2` supports: tensor(bool)

#### `IsNaN`
##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T2` supports: tensor(bool)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
  - [x] [x] [x] `T2` supports: tensor(bool)

##### Opset 20
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)
  - [x] [x] [x] `T2` supports: tensor(bool)

#### `LRN`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `alpha` (type: AttrType.FLOAT)
  - [x] [x] [x] `beta` (type: AttrType.FLOAT)
  - [x] [x] [x] `bias` (type: AttrType.FLOAT)
  - [x] [x] [x] `size` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `alpha` (type: AttrType.FLOAT)
  - [x] [x] [x] `beta` (type: AttrType.FLOAT)
  - [x] [x] [x] `bias` (type: AttrType.FLOAT)
  - [x] [x] [x] `size` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `LSTM`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `activation_alpha` (type: AttrType.FLOATS)
  - [x] [x] [x] `activation_beta` (type: AttrType.FLOATS)
  - [x] [x] [x] `activations` (type: AttrType.STRINGS)
  - [x] [x] [x] `clip` (type: AttrType.FLOAT)
  - [x] [x] [x] `direction` (type: AttrType.STRING)
  - [x] [x] [x] `hidden_size` (type: AttrType.INT)
  - [x] [x] [x] `input_forget` (type: AttrType.INT)
  - [x] [x] [x] `output_sequence` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `W`
  - [x] [x] [x] `R`
  - [x] [x] [x] `B`
  - [x] [x] [x] `sequence_lens`
  - [x] [x] [x] `initial_h`
  - [x] [x] [x] `initial_c`
  - [x] [x] [x] `P`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `Y_h`
  - [x] [x] [x] `Y_c`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T1` supports: tensor(int32)

##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `activation_alpha` (type: AttrType.FLOATS)
  - [x] [x] [x] `activation_beta` (type: AttrType.FLOATS)
  - [x] [x] [x] `activations` (type: AttrType.STRINGS)
  - [x] [x] [x] `clip` (type: AttrType.FLOAT)
  - [x] [x] [x] `direction` (type: AttrType.STRING)
  - [x] [x] [x] `hidden_size` (type: AttrType.INT)
  - [x] [x] [x] `input_forget` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `W`
  - [x] [x] [x] `R`
  - [x] [x] [x] `B`
  - [x] [x] [x] `sequence_lens`
  - [x] [x] [x] `initial_h`
  - [x] [x] [x] `initial_c`
  - [x] [x] [x] `P`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `Y_h`
  - [x] [x] [x] `Y_c`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T1` supports: tensor(int32)

##### Opset 14
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `activation_alpha` (type: AttrType.FLOATS)
  - [x] [x] [x] `activation_beta` (type: AttrType.FLOATS)
  - [x] [x] [x] `activations` (type: AttrType.STRINGS)
  - [x] [x] [x] `clip` (type: AttrType.FLOAT)
  - [x] [x] [x] `direction` (type: AttrType.STRING)
  - [x] [x] [x] `hidden_size` (type: AttrType.INT)
  - [x] [x] [x] `input_forget` (type: AttrType.INT)
  - [x] [x] [x] `layout` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `W`
  - [x] [x] [x] `R`
  - [x] [x] [x] `B`
  - [x] [x] [x] `sequence_lens`
  - [x] [x] [x] `initial_h`
  - [x] [x] [x] `initial_c`
  - [x] [x] [x] `P`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `Y_h`
  - [x] [x] [x] `Y_c`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T1` supports: tensor(int32)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `activation_alpha` (type: AttrType.FLOATS)
  - [x] [x] [x] `activation_beta` (type: AttrType.FLOATS)
  - [x] [x] [x] `activations` (type: AttrType.STRINGS)
  - [x] [x] [x] `clip` (type: AttrType.FLOAT)
  - [x] [x] [x] `direction` (type: AttrType.STRING)
  - [x] [x] [x] `hidden_size` (type: AttrType.INT)
  - [x] [x] [x] `input_forget` (type: AttrType.INT)
  - [x] [x] [x] `layout` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `W`
  - [x] [x] [x] `R`
  - [x] [x] [x] `B`
  - [x] [x] [x] `sequence_lens`
  - [x] [x] [x] `initial_h`
  - [x] [x] [x] `initial_c`
  - [x] [x] [x] `P`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `Y_h`
  - [x] [x] [x] `Y_c`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T1` supports: tensor(int32)

#### `LayerNormalization`
##### Opset 17
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `epsilon` (type: AttrType.FLOAT)
  - [x] [x] [x] `stash_type` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `Scale`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `Mean`
  - [x] [x] [x] `InvStdDev`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
  - [x] [x] [x] `U` supports: tensor(float), tensor(bfloat16)

#### `LeakyRelu`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `alpha` (type: AttrType.FLOAT)
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `alpha` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 16
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `alpha` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `Less`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `broadcast` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T1` supports: tensor(bool)

##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T1` supports: tensor(bool)

##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T1` supports: tensor(bool)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
  - [x] [x] [x] `T1` supports: tensor(bool)

#### `LessOrEqual`
##### Opset 12
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T1` supports: tensor(bool)

##### Opset 16
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
  - [x] [x] [x] `T1` supports: tensor(bool)

#### `Log`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `LogSoftmax`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `Loop`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `body` (type: AttrType.GRAPH)
- **Inputs:**
  - [x] [x] [x] `M`
  - [x] [x] [x] `cond`
  - [x] [x] [x] `v_initial`
- **Outputs:**
  - [x] [x] [x] `v_final_and_scan_outputs`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `I` supports: tensor(int64)
  - [x] [x] [x] `B` supports: tensor(bool)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `body` (type: AttrType.GRAPH)
- **Inputs:**
  - [x] [x] [x] `M`
  - [x] [x] [x] `cond`
  - [x] [x] [x] `v_initial`
- **Outputs:**
  - [x] [x] [x] `v_final_and_scan_outputs`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `I` supports: tensor(int64)
  - [x] [x] [x] `B` supports: tensor(bool)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `body` (type: AttrType.GRAPH)
- **Inputs:**
  - [x] [x] [x] `M`
  - [x] [x] [x] `cond`
  - [x] [x] [x] `v_initial`
- **Outputs:**
  - [x] [x] [x] `v_final_and_scan_outputs`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128))
  - [x] [x] [x] `I` supports: tensor(int64)
  - [x] [x] [x] `B` supports: tensor(bool)

##### Opset 16
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `body` (type: AttrType.GRAPH)
- **Inputs:**
  - [x] [x] [x] `M`
  - [x] [x] [x] `cond`
  - [x] [x] [x] `v_initial`
- **Outputs:**
  - [x] [x] [x] `v_final_and_scan_outputs`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(bfloat16)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128)), optional(seq(tensor(uint8))), optional(seq(tensor(uint16))), optional(seq(tensor(uint32))), optional(seq(tensor(uint64))), optional(seq(tensor(int8))), optional(seq(tensor(int16))), optional(seq(tensor(int32))), optional(seq(tensor(int64))), optional(seq(tensor(bfloat16))), optional(seq(tensor(float16))), optional(seq(tensor(float))), optional(seq(tensor(double))), optional(seq(tensor(string))), optional(seq(tensor(bool))), optional(seq(tensor(complex64))), optional(seq(tensor(complex128))), optional(tensor(uint8)), optional(tensor(uint16)), optional(tensor(uint32)), optional(tensor(uint64)), optional(tensor(int8)), optional(tensor(int16)), optional(tensor(int32)), optional(tensor(int64)), optional(tensor(bfloat16)), optional(tensor(float16)), optional(tensor(float)), optional(tensor(double)), optional(tensor(string)), optional(tensor(bool)), optional(tensor(complex64)), optional(tensor(complex128))
  - [x] [x] [x] `I` supports: tensor(int64)
  - [x] [x] [x] `B` supports: tensor(bool)

##### Opset 19
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `body` (type: AttrType.GRAPH)
- **Inputs:**
  - [x] [x] [x] `M`
  - [x] [x] [x] `cond`
  - [x] [x] [x] `v_initial`
- **Outputs:**
  - [x] [x] [x] `v_final_and_scan_outputs`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(bfloat16)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128)), seq(tensor(float8e4m3fn)), seq(tensor(float8e4m3fnuz)), seq(tensor(float8e5m2)), seq(tensor(float8e5m2fnuz)), optional(seq(tensor(uint8))), optional(seq(tensor(uint16))), optional(seq(tensor(uint32))), optional(seq(tensor(uint64))), optional(seq(tensor(int8))), optional(seq(tensor(int16))), optional(seq(tensor(int32))), optional(seq(tensor(int64))), optional(seq(tensor(bfloat16))), optional(seq(tensor(float16))), optional(seq(tensor(float))), optional(seq(tensor(double))), optional(seq(tensor(string))), optional(seq(tensor(bool))), optional(seq(tensor(complex64))), optional(seq(tensor(complex128))), optional(tensor(uint8)), optional(tensor(uint16)), optional(tensor(uint32)), optional(tensor(uint64)), optional(tensor(int8)), optional(tensor(int16)), optional(tensor(int32)), optional(tensor(int64)), optional(tensor(bfloat16)), optional(tensor(float16)), optional(tensor(float)), optional(tensor(double)), optional(tensor(string)), optional(tensor(bool)), optional(tensor(complex64)), optional(tensor(complex128)), optional(tensor(float8e4m3fn)), optional(tensor(float8e4m3fnuz)), optional(tensor(float8e5m2)), optional(tensor(float8e5m2fnuz))
  - [x] [x] [x] `I` supports: tensor(int64)
  - [x] [x] [x] `B` supports: tensor(bool)

##### Opset 21
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `body` (type: AttrType.GRAPH)
- **Inputs:**
  - [x] [x] [x] `M`
  - [x] [x] [x] `cond`
  - [x] [x] [x] `v_initial`
- **Outputs:**
  - [x] [x] [x] `v_final_and_scan_outputs`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(bfloat16)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128)), seq(tensor(float8e4m3fn)), seq(tensor(float8e4m3fnuz)), seq(tensor(float8e5m2)), seq(tensor(float8e5m2fnuz)), seq(tensor(uint4)), seq(tensor(int4)), optional(seq(tensor(uint8))), optional(seq(tensor(uint16))), optional(seq(tensor(uint32))), optional(seq(tensor(uint64))), optional(seq(tensor(int8))), optional(seq(tensor(int16))), optional(seq(tensor(int32))), optional(seq(tensor(int64))), optional(seq(tensor(bfloat16))), optional(seq(tensor(float16))), optional(seq(tensor(float))), optional(seq(tensor(double))), optional(seq(tensor(string))), optional(seq(tensor(bool))), optional(seq(tensor(complex64))), optional(seq(tensor(complex128))), optional(tensor(uint8)), optional(tensor(uint16)), optional(tensor(uint32)), optional(tensor(uint64)), optional(tensor(int8)), optional(tensor(int16)), optional(tensor(int32)), optional(tensor(int64)), optional(tensor(bfloat16)), optional(tensor(float16)), optional(tensor(float)), optional(tensor(double)), optional(tensor(string)), optional(tensor(bool)), optional(tensor(complex64)), optional(tensor(complex128)), optional(tensor(float8e4m3fn)), optional(tensor(float8e4m3fnuz)), optional(tensor(float8e5m2)), optional(tensor(float8e5m2fnuz)), optional(tensor(uint4)), optional(tensor(int4))
  - [x] [x] [x] `I` supports: tensor(int64)
  - [x] [x] [x] `B` supports: tensor(bool)

##### Opset 23
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `body` (type: AttrType.GRAPH)
- **Inputs:**
  - [x] [x] [x] `M`
  - [x] [x] [x] `cond`
  - [x] [x] [x] `v_initial`
- **Outputs:**
  - [x] [x] [x] `v_final_and_scan_outputs`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1), seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(bfloat16)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128)), seq(tensor(float8e4m3fn)), seq(tensor(float8e4m3fnuz)), seq(tensor(float8e5m2)), seq(tensor(float8e5m2fnuz)), seq(tensor(uint4)), seq(tensor(int4)), seq(tensor(float4e2m1)), optional(seq(tensor(uint8))), optional(seq(tensor(uint16))), optional(seq(tensor(uint32))), optional(seq(tensor(uint64))), optional(seq(tensor(int8))), optional(seq(tensor(int16))), optional(seq(tensor(int32))), optional(seq(tensor(int64))), optional(seq(tensor(bfloat16))), optional(seq(tensor(float16))), optional(seq(tensor(float))), optional(seq(tensor(double))), optional(seq(tensor(string))), optional(seq(tensor(bool))), optional(seq(tensor(complex64))), optional(seq(tensor(complex128))), optional(tensor(uint8)), optional(tensor(uint16)), optional(tensor(uint32)), optional(tensor(uint64)), optional(tensor(int8)), optional(tensor(int16)), optional(tensor(int32)), optional(tensor(int64)), optional(tensor(bfloat16)), optional(tensor(float16)), optional(tensor(float)), optional(tensor(double)), optional(tensor(string)), optional(tensor(bool)), optional(tensor(complex64)), optional(tensor(complex128)), optional(tensor(float8e4m3fn)), optional(tensor(float8e4m3fnuz)), optional(tensor(float8e5m2)), optional(tensor(float8e5m2fnuz)), optional(tensor(uint4)), optional(tensor(int4)), optional(tensor(float4e2m1))
  - [x] [x] [x] `I` supports: tensor(int64)
  - [x] [x] [x] `B` supports: tensor(bool)

##### Opset 24
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `body` (type: AttrType.GRAPH)
- **Inputs:**
  - [x] [x] [x] `M`
  - [x] [x] [x] `cond`
  - [x] [x] [x] `v_initial`
- **Outputs:**
  - [x] [x] [x] `v_final_and_scan_outputs`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1), tensor(float8e8m0), seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(bfloat16)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128)), seq(tensor(float8e4m3fn)), seq(tensor(float8e4m3fnuz)), seq(tensor(float8e5m2)), seq(tensor(float8e5m2fnuz)), seq(tensor(uint4)), seq(tensor(int4)), seq(tensor(float4e2m1)), seq(tensor(float8e8m0)), optional(seq(tensor(uint8))), optional(seq(tensor(uint16))), optional(seq(tensor(uint32))), optional(seq(tensor(uint64))), optional(seq(tensor(int8))), optional(seq(tensor(int16))), optional(seq(tensor(int32))), optional(seq(tensor(int64))), optional(seq(tensor(bfloat16))), optional(seq(tensor(float16))), optional(seq(tensor(float))), optional(seq(tensor(double))), optional(seq(tensor(string))), optional(seq(tensor(bool))), optional(seq(tensor(complex64))), optional(seq(tensor(complex128))), optional(tensor(uint8)), optional(tensor(uint16)), optional(tensor(uint32)), optional(tensor(uint64)), optional(tensor(int8)), optional(tensor(int16)), optional(tensor(int32)), optional(tensor(int64)), optional(tensor(bfloat16)), optional(tensor(float16)), optional(tensor(float)), optional(tensor(double)), optional(tensor(string)), optional(tensor(bool)), optional(tensor(complex64)), optional(tensor(complex128)), optional(tensor(float8e4m3fn)), optional(tensor(float8e4m3fnuz)), optional(tensor(float8e5m2)), optional(tensor(float8e5m2fnuz)), optional(tensor(uint4)), optional(tensor(int4)), optional(tensor(float4e2m1)), optional(tensor(float8e8m0))
  - [x] [x] [x] `I` supports: tensor(int64)
  - [x] [x] [x] `B` supports: tensor(bool)

#### `LpNormalization`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `p` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `p` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `LpPool`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `auto_pad` (type: AttrType.STRING)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `p` (type: AttrType.FLOAT)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 2
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `auto_pad` (type: AttrType.STRING)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `p` (type: AttrType.INT)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `auto_pad` (type: AttrType.STRING)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `p` (type: AttrType.INT)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 18
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `auto_pad` (type: AttrType.STRING)
  - [x] [x] [x] `ceil_mode` (type: AttrType.INT)
  - [x] [x] [x] `dilations` (type: AttrType.INTS)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `p` (type: AttrType.INT)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `auto_pad` (type: AttrType.STRING)
  - [x] [x] [x] `ceil_mode` (type: AttrType.INT)
  - [x] [x] [x] `dilations` (type: AttrType.INTS)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `p` (type: AttrType.INT)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `MatMul`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(uint32), tensor(uint64), tensor(int32), tensor(int64)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(bfloat16)

#### `MatMulInteger`
##### Opset 10
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
  - [x] [x] [x] `a_zero_point`
  - [x] [x] [x] `b_zero_point`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(int8), tensor(uint8)
  - [x] [x] [x] `T2` supports: tensor(int8), tensor(uint8)
  - [x] [x] [x] `T3` supports: tensor(int32)

#### `Max`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `data_0`
- **Outputs:**
  - [x] [x] [x] `max`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data_0`
- **Outputs:**
  - [x] [x] [x] `max`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 8
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data_0`
- **Outputs:**
  - [x] [x] [x] `max`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 12
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data_0`
- **Outputs:**
  - [x] [x] [x] `max`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data_0`
- **Outputs:**
  - [x] [x] [x] `max`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `MaxPool`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `auto_pad` (type: AttrType.STRING)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 8
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `auto_pad` (type: AttrType.STRING)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `storage_order` (type: AttrType.INT)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `Indices`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `I` supports: tensor(int64)

##### Opset 10
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `auto_pad` (type: AttrType.STRING)
  - [x] [x] [x] `ceil_mode` (type: AttrType.INT)
  - [x] [x] [x] `dilations` (type: AttrType.INTS)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `storage_order` (type: AttrType.INT)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `Indices`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `I` supports: tensor(int64)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `auto_pad` (type: AttrType.STRING)
  - [x] [x] [x] `ceil_mode` (type: AttrType.INT)
  - [x] [x] [x] `dilations` (type: AttrType.INTS)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `storage_order` (type: AttrType.INT)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `Indices`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `I` supports: tensor(int64)

##### Opset 12
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `auto_pad` (type: AttrType.STRING)
  - [x] [x] [x] `ceil_mode` (type: AttrType.INT)
  - [x] [x] [x] `dilations` (type: AttrType.INTS)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `storage_order` (type: AttrType.INT)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `Indices`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(uint8)
  - [x] [x] [x] `I` supports: tensor(int64)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `auto_pad` (type: AttrType.STRING)
  - [x] [x] [x] `ceil_mode` (type: AttrType.INT)
  - [x] [x] [x] `dilations` (type: AttrType.INTS)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `storage_order` (type: AttrType.INT)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `Indices`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(uint8)
  - [x] [x] [x] `I` supports: tensor(int64)

#### `MaxRoiPool`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `pooled_shape` (type: AttrType.INTS)
  - [x] [x] [x] `spatial_scale` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `rois`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `pooled_shape` (type: AttrType.INTS)
  - [x] [x] [x] `spatial_scale` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `rois`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `MaxUnpool`
##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `I`
  - [x] [x] [x] `output_shape`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T2` supports: tensor(int64)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `I`
  - [x] [x] [x] `output_shape`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T2` supports: tensor(int64)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `I`
  - [x] [x] [x] `output_shape`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T2` supports: tensor(int64)

#### `Mean`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `data_0`
- **Outputs:**
  - [x] [x] [x] `mean`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data_0`
- **Outputs:**
  - [x] [x] [x] `mean`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 8
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data_0`
- **Outputs:**
  - [x] [x] [x] `mean`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data_0`
- **Outputs:**
  - [x] [x] [x] `mean`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `MeanVarianceNormalization`
##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `MelWeightMatrix`
##### Opset 17
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `output_datatype` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `num_mel_bins`
  - [x] [x] [x] `dft_length`
  - [x] [x] [x] `sample_rate`
  - [x] [x] [x] `lower_edge_hertz`
  - [x] [x] [x] `upper_edge_hertz`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(int32), tensor(int64)
  - [x] [x] [x] `T2` supports: tensor(float), tensor(float16), tensor(double), tensor(bfloat16)
  - [x] [x] [x] `T3` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `Min`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `data_0`
- **Outputs:**
  - [x] [x] [x] `min`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data_0`
- **Outputs:**
  - [x] [x] [x] `min`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 8
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data_0`
- **Outputs:**
  - [x] [x] [x] `min`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 12
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data_0`
- **Outputs:**
  - [x] [x] [x] `min`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data_0`
- **Outputs:**
  - [x] [x] [x] `min`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `Mish`
##### Opset 18
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `Mod`
##### Opset 10
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `fmod` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `fmod` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `Mul`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `broadcast` (type: AttrType.INT)
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `broadcast` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

##### Opset 14
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `Multinomial`
##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `dtype` (type: AttrType.INT)
  - [x] [x] [x] `sample_size` (type: AttrType.INT)
  - [x] [x] [x] `seed` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T2` supports: tensor(int32), tensor(int64)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `dtype` (type: AttrType.INT)
  - [x] [x] [x] `sample_size` (type: AttrType.INT)
  - [x] [x] [x] `seed` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T2` supports: tensor(int32), tensor(int64)

#### `Neg`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float), tensor(int32), tensor(int8), tensor(int16), tensor(int64), tensor(float16), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float), tensor(int32), tensor(int8), tensor(int16), tensor(int64), tensor(float16), tensor(double), tensor(bfloat16)

#### `NegativeLogLikelihoodLoss`
##### Opset 12
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `ignore_index` (type: AttrType.INT)
  - [x] [x] [x] `reduction` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `target`
  - [x] [x] [x] `weight`
- **Outputs:**
  - [x] [x] [x] `loss`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `Tind` supports: tensor(int32), tensor(int64)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `ignore_index` (type: AttrType.INT)
  - [x] [x] [x] `reduction` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `target`
  - [x] [x] [x] `weight`
- **Outputs:**
  - [x] [x] [x] `loss`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `Tind` supports: tensor(int32), tensor(int64)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `ignore_index` (type: AttrType.INT)
  - [x] [x] [x] `reduction` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `target`
  - [x] [x] [x] `weight`
- **Outputs:**
  - [x] [x] [x] `loss`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `Tind` supports: tensor(int32), tensor(int64)

#### `NonMaxSuppression`
##### Opset 10
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `center_point_box` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `boxes`
  - [x] [x] [x] `scores`
  - [x] [x] [x] `max_output_boxes_per_class`
  - [x] [x] [x] `iou_threshold`
  - [x] [x] [x] `score_threshold`
- **Outputs:**
  - [x] [x] [x] `selected_indices`

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `center_point_box` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `boxes`
  - [x] [x] [x] `scores`
  - [x] [x] [x] `max_output_boxes_per_class`
  - [x] [x] [x] `iou_threshold`
  - [x] [x] [x] `score_threshold`
- **Outputs:**
  - [x] [x] [x] `selected_indices`

#### `NonZero`
##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

#### `Not`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bool)

#### `OneHot`
##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `indices`
  - [x] [x] [x] `depth`
  - [x] [x] [x] `values`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T2` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T3` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `indices`
  - [x] [x] [x] `depth`
  - [x] [x] [x] `values`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T2` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T3` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

#### `Optional`
##### Opset 15
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `type` (type: AttrType.TYPE_PROTO)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128))
  - [x] [x] [x] `O` supports: optional(seq(tensor(uint8))), optional(seq(tensor(uint16))), optional(seq(tensor(uint32))), optional(seq(tensor(uint64))), optional(seq(tensor(int8))), optional(seq(tensor(int16))), optional(seq(tensor(int32))), optional(seq(tensor(int64))), optional(seq(tensor(float16))), optional(seq(tensor(float))), optional(seq(tensor(double))), optional(seq(tensor(string))), optional(seq(tensor(bool))), optional(seq(tensor(complex64))), optional(seq(tensor(complex128))), optional(tensor(uint8)), optional(tensor(uint16)), optional(tensor(uint32)), optional(tensor(uint64)), optional(tensor(int8)), optional(tensor(int16)), optional(tensor(int32)), optional(tensor(int64)), optional(tensor(float16)), optional(tensor(float)), optional(tensor(double)), optional(tensor(string)), optional(tensor(bool)), optional(tensor(complex64)), optional(tensor(complex128))

#### `OptionalGetElement`
##### Opset 15
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `O` supports: optional(seq(tensor(uint8))), optional(seq(tensor(uint16))), optional(seq(tensor(uint32))), optional(seq(tensor(uint64))), optional(seq(tensor(int8))), optional(seq(tensor(int16))), optional(seq(tensor(int32))), optional(seq(tensor(int64))), optional(seq(tensor(float16))), optional(seq(tensor(float))), optional(seq(tensor(double))), optional(seq(tensor(string))), optional(seq(tensor(bool))), optional(seq(tensor(complex64))), optional(seq(tensor(complex128))), optional(tensor(uint8)), optional(tensor(uint16)), optional(tensor(uint32)), optional(tensor(uint64)), optional(tensor(int8)), optional(tensor(int16)), optional(tensor(int32)), optional(tensor(int64)), optional(tensor(float16)), optional(tensor(float)), optional(tensor(double)), optional(tensor(string)), optional(tensor(bool)), optional(tensor(complex64)), optional(tensor(complex128))
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128))

##### Opset 18
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `O` supports: optional(seq(tensor(uint8))), optional(seq(tensor(uint16))), optional(seq(tensor(uint32))), optional(seq(tensor(uint64))), optional(seq(tensor(int8))), optional(seq(tensor(int16))), optional(seq(tensor(int32))), optional(seq(tensor(int64))), optional(seq(tensor(float16))), optional(seq(tensor(float))), optional(seq(tensor(double))), optional(seq(tensor(string))), optional(seq(tensor(bool))), optional(seq(tensor(complex64))), optional(seq(tensor(complex128))), optional(tensor(uint8)), optional(tensor(uint16)), optional(tensor(uint32)), optional(tensor(uint64)), optional(tensor(int8)), optional(tensor(int16)), optional(tensor(int32)), optional(tensor(int64)), optional(tensor(float16)), optional(tensor(float)), optional(tensor(double)), optional(tensor(string)), optional(tensor(bool)), optional(tensor(complex64)), optional(tensor(complex128)), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128))
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128))

#### `OptionalHasElement`
##### Opset 15
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `O` supports: optional(seq(tensor(uint8))), optional(seq(tensor(uint16))), optional(seq(tensor(uint32))), optional(seq(tensor(uint64))), optional(seq(tensor(int8))), optional(seq(tensor(int16))), optional(seq(tensor(int32))), optional(seq(tensor(int64))), optional(seq(tensor(float16))), optional(seq(tensor(float))), optional(seq(tensor(double))), optional(seq(tensor(string))), optional(seq(tensor(bool))), optional(seq(tensor(complex64))), optional(seq(tensor(complex128))), optional(tensor(uint8)), optional(tensor(uint16)), optional(tensor(uint32)), optional(tensor(uint64)), optional(tensor(int8)), optional(tensor(int16)), optional(tensor(int32)), optional(tensor(int64)), optional(tensor(float16)), optional(tensor(float)), optional(tensor(double)), optional(tensor(string)), optional(tensor(bool)), optional(tensor(complex64)), optional(tensor(complex128))
  - [x] [x] [x] `B` supports: tensor(bool)

##### Opset 18
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `O` supports: optional(seq(tensor(uint8))), optional(seq(tensor(uint16))), optional(seq(tensor(uint32))), optional(seq(tensor(uint64))), optional(seq(tensor(int8))), optional(seq(tensor(int16))), optional(seq(tensor(int32))), optional(seq(tensor(int64))), optional(seq(tensor(float16))), optional(seq(tensor(float))), optional(seq(tensor(double))), optional(seq(tensor(string))), optional(seq(tensor(bool))), optional(seq(tensor(complex64))), optional(seq(tensor(complex128))), optional(tensor(uint8)), optional(tensor(uint16)), optional(tensor(uint32)), optional(tensor(uint64)), optional(tensor(int8)), optional(tensor(int16)), optional(tensor(int32)), optional(tensor(int64)), optional(tensor(float16)), optional(tensor(float)), optional(tensor(double)), optional(tensor(string)), optional(tensor(bool)), optional(tensor(complex64)), optional(tensor(complex128)), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128))
  - [x] [x] [x] `B` supports: tensor(bool)

#### `Or`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `broadcast` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bool)
  - [x] [x] [x] `T1` supports: tensor(bool)

##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bool)
  - [x] [x] [x] `T1` supports: tensor(bool)

#### `PRelu`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `slope`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `slope`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `slope`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `slope`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(uint32), tensor(uint64), tensor(int32), tensor(int64)

##### Opset 16
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `slope`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(uint32), tensor(uint64), tensor(int32), tensor(int64)

#### `Pad`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `mode` (type: AttrType.STRING)
  - [x] [x] [x] `paddings` (type: AttrType.INTS)
  - [x] [x] [x] `value` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 2
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `mode` (type: AttrType.STRING)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `value` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `mode` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `pads`
  - [x] [x] [x] `constant_value`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `mode` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `pads`
  - [x] [x] [x] `constant_value`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 18
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `mode` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `pads`
  - [x] [x] [x] `constant_value`
  - [x] [x] [x] `axes`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `Tind` supports: tensor(int32), tensor(int64)

##### Opset 19
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `mode` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `pads`
  - [x] [x] [x] `constant_value`
  - [x] [x] [x] `axes`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `Tind` supports: tensor(int32), tensor(int64)

##### Opset 21
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `mode` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `pads`
  - [x] [x] [x] `constant_value`
  - [x] [x] [x] `axes`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4)
  - [x] [x] [x] `Tind` supports: tensor(int32), tensor(int64)

##### Opset 23
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `mode` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `pads`
  - [x] [x] [x] `constant_value`
  - [x] [x] [x] `axes`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1)
  - [x] [x] [x] `Tind` supports: tensor(int32), tensor(int64)

##### Opset 24
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `mode` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `pads`
  - [x] [x] [x] `constant_value`
  - [x] [x] [x] `axes`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1), tensor(float8e8m0)
  - [x] [x] [x] `Tind` supports: tensor(int32), tensor(int64)

#### `Pow`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `broadcast` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `Y`
- **Outputs:**
  - [x] [x] [x] `Z`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `Y`
- **Outputs:**
  - [x] [x] [x] `Z`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 12
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `Y`
- **Outputs:**
  - [x] [x] [x] `Z`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T1` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `Y`
- **Outputs:**
  - [x] [x] [x] `Z`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
  - [x] [x] [x] `T1` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 15
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `Y`
- **Outputs:**
  - [x] [x] [x] `Z`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
  - [x] [x] [x] `T1` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `QLinearConv`
##### Opset 10
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `auto_pad` (type: AttrType.STRING)
  - [x] [x] [x] `dilations` (type: AttrType.INTS)
  - [x] [x] [x] `group` (type: AttrType.INT)
  - [x] [x] [x] `kernel_shape` (type: AttrType.INTS)
  - [x] [x] [x] `pads` (type: AttrType.INTS)
  - [x] [x] [x] `strides` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `x`
  - [x] [x] [x] `x_scale`
  - [x] [x] [x] `x_zero_point`
  - [x] [x] [x] `w`
  - [x] [x] [x] `w_scale`
  - [x] [x] [x] `w_zero_point`
  - [x] [x] [x] `y_scale`
  - [x] [x] [x] `y_zero_point`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(int8), tensor(uint8)
  - [x] [x] [x] `T2` supports: tensor(int8), tensor(uint8)
  - [x] [x] [x] `T3` supports: tensor(int8), tensor(uint8)
  - [x] [x] [x] `T4` supports: tensor(int32)

#### `QLinearMatMul`
##### Opset 10
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `a`
  - [x] [x] [x] `a_scale`
  - [x] [x] [x] `a_zero_point`
  - [x] [x] [x] `b`
  - [x] [x] [x] `b_scale`
  - [x] [x] [x] `b_zero_point`
  - [x] [x] [x] `y_scale`
  - [x] [x] [x] `y_zero_point`
- **Outputs:**
  - [x] [x] [x] `y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(int8), tensor(uint8)
  - [x] [x] [x] `T2` supports: tensor(int8), tensor(uint8)
  - [x] [x] [x] `T3` supports: tensor(int8), tensor(uint8)

##### Opset 21
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `a`
  - [x] [x] [x] `a_scale`
  - [x] [x] [x] `a_zero_point`
  - [x] [x] [x] `b`
  - [x] [x] [x] `b_scale`
  - [x] [x] [x] `b_zero_point`
  - [x] [x] [x] `y_scale`
  - [x] [x] [x] `y_zero_point`
- **Outputs:**
  - [x] [x] [x] `y`
- **Type Constraints:**
  - [x] [x] [x] `TS` supports: tensor(float), tensor(float16), tensor(bfloat16)
  - [x] [x] [x] `T1` supports: tensor(int8), tensor(uint8), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)
  - [x] [x] [x] `T2` supports: tensor(int8), tensor(uint8), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)
  - [x] [x] [x] `T3` supports: tensor(int8), tensor(uint8), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)

#### `QuantizeLinear`
##### Opset 10
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `x`
  - [x] [x] [x] `y_scale`
  - [x] [x] [x] `y_zero_point`
- **Outputs:**
  - [x] [x] [x] `y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float), tensor(int32)
  - [x] [x] [x] `T2` supports: tensor(int8), tensor(uint8)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `x`
  - [x] [x] [x] `y_scale`
  - [x] [x] [x] `y_zero_point`
- **Outputs:**
  - [x] [x] [x] `y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float), tensor(int32)
  - [x] [x] [x] `T2` supports: tensor(int8), tensor(uint8)

##### Opset 19
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `saturate` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `x`
  - [x] [x] [x] `y_scale`
  - [x] [x] [x] `y_zero_point`
- **Outputs:**
  - [x] [x] [x] `y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float), tensor(float16), tensor(bfloat16), tensor(int32)
  - [x] [x] [x] `T2` supports: tensor(int8), tensor(uint8), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)

##### Opset 21
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `block_size` (type: AttrType.INT)
  - [x] [x] [x] `output_dtype` (type: AttrType.INT)
  - [x] [x] [x] `saturate` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `x`
  - [x] [x] [x] `y_scale`
  - [x] [x] [x] `y_zero_point`
- **Outputs:**
  - [x] [x] [x] `y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float), tensor(float16), tensor(bfloat16), tensor(int32)
  - [x] [x] [x] `T2` supports: tensor(int8), tensor(uint8), tensor(int16), tensor(uint16), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4)

##### Opset 23
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `block_size` (type: AttrType.INT)
  - [x] [x] [x] `output_dtype` (type: AttrType.INT)
  - [x] [x] [x] `precision` (type: AttrType.INT)
  - [x] [x] [x] `saturate` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `x`
  - [x] [x] [x] `y_scale`
  - [x] [x] [x] `y_zero_point`
- **Outputs:**
  - [x] [x] [x] `y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float), tensor(float16), tensor(bfloat16), tensor(int32)
  - [x] [x] [x] `T2` supports: tensor(float), tensor(float16), tensor(bfloat16), tensor(int32)
  - [x] [x] [x] `T3` supports: tensor(int8), tensor(uint8), tensor(int16), tensor(uint16), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1)

##### Opset 24
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `block_size` (type: AttrType.INT)
  - [x] [x] [x] `output_dtype` (type: AttrType.INT)
  - [x] [x] [x] `precision` (type: AttrType.INT)
  - [x] [x] [x] `saturate` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `x`
  - [x] [x] [x] `y_scale`
  - [x] [x] [x] `y_zero_point`
- **Outputs:**
  - [x] [x] [x] `y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float), tensor(float16), tensor(bfloat16), tensor(int32)
  - [x] [x] [x] `T2` supports: tensor(float), tensor(float16), tensor(bfloat16), tensor(int32), tensor(float8e8m0)
  - [x] [x] [x] `T3` supports: tensor(int8), tensor(uint8), tensor(int16), tensor(uint16), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1)

#### `RMSNormalization`
##### Opset 23
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `epsilon` (type: AttrType.FLOAT)
  - [x] [x] [x] `stash_type` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `scale`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
  - [x] [x] [x] `V` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `RNN`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `activation_alpha` (type: AttrType.FLOATS)
  - [x] [x] [x] `activation_beta` (type: AttrType.FLOATS)
  - [x] [x] [x] `activations` (type: AttrType.STRINGS)
  - [x] [x] [x] `clip` (type: AttrType.FLOAT)
  - [x] [x] [x] `direction` (type: AttrType.STRING)
  - [x] [x] [x] `hidden_size` (type: AttrType.INT)
  - [x] [x] [x] `output_sequence` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `W`
  - [x] [x] [x] `R`
  - [x] [x] [x] `B`
  - [x] [x] [x] `sequence_lens`
  - [x] [x] [x] `initial_h`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `Y_h`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T1` supports: tensor(int32)

##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `activation_alpha` (type: AttrType.FLOATS)
  - [x] [x] [x] `activation_beta` (type: AttrType.FLOATS)
  - [x] [x] [x] `activations` (type: AttrType.STRINGS)
  - [x] [x] [x] `clip` (type: AttrType.FLOAT)
  - [x] [x] [x] `direction` (type: AttrType.STRING)
  - [x] [x] [x] `hidden_size` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `W`
  - [x] [x] [x] `R`
  - [x] [x] [x] `B`
  - [x] [x] [x] `sequence_lens`
  - [x] [x] [x] `initial_h`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `Y_h`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T1` supports: tensor(int32)

##### Opset 14
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `activation_alpha` (type: AttrType.FLOATS)
  - [x] [x] [x] `activation_beta` (type: AttrType.FLOATS)
  - [x] [x] [x] `activations` (type: AttrType.STRINGS)
  - [x] [x] [x] `clip` (type: AttrType.FLOAT)
  - [x] [x] [x] `direction` (type: AttrType.STRING)
  - [x] [x] [x] `hidden_size` (type: AttrType.INT)
  - [x] [x] [x] `layout` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `W`
  - [x] [x] [x] `R`
  - [x] [x] [x] `B`
  - [x] [x] [x] `sequence_lens`
  - [x] [x] [x] `initial_h`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `Y_h`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T1` supports: tensor(int32)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `activation_alpha` (type: AttrType.FLOATS)
  - [x] [x] [x] `activation_beta` (type: AttrType.FLOATS)
  - [x] [x] [x] `activations` (type: AttrType.STRINGS)
  - [x] [x] [x] `clip` (type: AttrType.FLOAT)
  - [x] [x] [x] `direction` (type: AttrType.STRING)
  - [x] [x] [x] `hidden_size` (type: AttrType.INT)
  - [x] [x] [x] `layout` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `W`
  - [x] [x] [x] `R`
  - [x] [x] [x] `B`
  - [x] [x] [x] `sequence_lens`
  - [x] [x] [x] `initial_h`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `Y_h`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T1` supports: tensor(int32)

#### `RandomNormal`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `dtype` (type: AttrType.INT)
  - [x] [x] [x] `mean` (type: AttrType.FLOAT)
  - [x] [x] [x] `scale` (type: AttrType.FLOAT)
  - [x] [x] [x] `seed` (type: AttrType.FLOAT)
  - [x] [x] [x] `shape` (type: AttrType.INTS)
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `dtype` (type: AttrType.INT)
  - [x] [x] [x] `mean` (type: AttrType.FLOAT)
  - [x] [x] [x] `scale` (type: AttrType.FLOAT)
  - [x] [x] [x] `seed` (type: AttrType.FLOAT)
  - [x] [x] [x] `shape` (type: AttrType.INTS)
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `RandomNormalLike`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `dtype` (type: AttrType.INT)
  - [x] [x] [x] `mean` (type: AttrType.FLOAT)
  - [x] [x] [x] `scale` (type: AttrType.FLOAT)
  - [x] [x] [x] `seed` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `T2` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `dtype` (type: AttrType.INT)
  - [x] [x] [x] `mean` (type: AttrType.FLOAT)
  - [x] [x] [x] `scale` (type: AttrType.FLOAT)
  - [x] [x] [x] `seed` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `T2` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `RandomUniform`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `dtype` (type: AttrType.INT)
  - [x] [x] [x] `high` (type: AttrType.FLOAT)
  - [x] [x] [x] `low` (type: AttrType.FLOAT)
  - [x] [x] [x] `seed` (type: AttrType.FLOAT)
  - [x] [x] [x] `shape` (type: AttrType.INTS)
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `dtype` (type: AttrType.INT)
  - [x] [x] [x] `high` (type: AttrType.FLOAT)
  - [x] [x] [x] `low` (type: AttrType.FLOAT)
  - [x] [x] [x] `seed` (type: AttrType.FLOAT)
  - [x] [x] [x] `shape` (type: AttrType.INTS)
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `RandomUniformLike`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `dtype` (type: AttrType.INT)
  - [x] [x] [x] `high` (type: AttrType.FLOAT)
  - [x] [x] [x] `low` (type: AttrType.FLOAT)
  - [x] [x] [x] `seed` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `T2` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `dtype` (type: AttrType.INT)
  - [x] [x] [x] `high` (type: AttrType.FLOAT)
  - [x] [x] [x] `low` (type: AttrType.FLOAT)
  - [x] [x] [x] `seed` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `T2` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `Range`
##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `start`
  - [x] [x] [x] `limit`
  - [x] [x] [x] `delta`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float), tensor(double), tensor(int16), tensor(int32), tensor(int64)

#### `Reciprocal`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `ReduceL1`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

##### Opset 18
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
  - [x] [x] [x] `noop_with_empty_axes` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `axes`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `ReduceL2`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

##### Opset 18
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
  - [x] [x] [x] `noop_with_empty_axes` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `axes`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `ReduceLogSum`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

##### Opset 18
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
  - [x] [x] [x] `noop_with_empty_axes` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `axes`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `ReduceLogSumExp`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

##### Opset 18
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
  - [x] [x] [x] `noop_with_empty_axes` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `axes`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `ReduceMax`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 12
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(uint8), tensor(int8)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16), tensor(uint8), tensor(int8)

##### Opset 18
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
  - [x] [x] [x] `noop_with_empty_axes` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `axes`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16), tensor(uint8), tensor(int8)

##### Opset 20
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
  - [x] [x] [x] `noop_with_empty_axes` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `axes`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16), tensor(uint8), tensor(int8), tensor(bool)

#### `ReduceMean`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

##### Opset 18
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
  - [x] [x] [x] `noop_with_empty_axes` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `axes`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `ReduceMin`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 12
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(uint8), tensor(int8)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16), tensor(uint8), tensor(int8)

##### Opset 18
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
  - [x] [x] [x] `noop_with_empty_axes` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `axes`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16), tensor(uint8), tensor(int8)

##### Opset 20
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
  - [x] [x] [x] `noop_with_empty_axes` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `axes`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16), tensor(uint8), tensor(int8), tensor(bool)

#### `ReduceProd`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

##### Opset 18
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
  - [x] [x] [x] `noop_with_empty_axes` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `axes`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `ReduceSum`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
  - [x] [x] [x] `noop_with_empty_axes` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `axes`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `ReduceSumSquare`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

##### Opset 18
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
  - [x] [x] [x] `noop_with_empty_axes` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `axes`
- **Outputs:**
  - [x] [x] [x] `reduced`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `RegexFullMatch`
##### Opset 20
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `pattern` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(string)
  - [x] [x] [x] `T2` supports: tensor(bool)

#### `Relu`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

##### Opset 14
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float), tensor(int32), tensor(int8), tensor(int16), tensor(int64), tensor(float16), tensor(double), tensor(bfloat16)

#### `Reshape`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
  - [x] [x] [x] `shape` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `reshaped`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 5
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `shape`
- **Outputs:**
  - [x] [x] [x] `reshaped`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `shape`
- **Outputs:**
  - [x] [x] [x] `reshaped`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 14
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `allowzero` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `shape`
- **Outputs:**
  - [x] [x] [x] `reshaped`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 19
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `allowzero` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `shape`
- **Outputs:**
  - [x] [x] [x] `reshaped`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)

##### Opset 21
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `allowzero` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `shape`
- **Outputs:**
  - [x] [x] [x] `reshaped`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4)

##### Opset 23
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `allowzero` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `shape`
- **Outputs:**
  - [x] [x] [x] `reshaped`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1)

##### Opset 24
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `allowzero` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `shape`
- **Outputs:**
  - [x] [x] [x] `reshaped`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1), tensor(float8e8m0)

#### `Resize`
##### Opset 10
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `mode` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `scales`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `coordinate_transformation_mode` (type: AttrType.STRING)
  - [x] [x] [x] `cubic_coeff_a` (type: AttrType.FLOAT)
  - [x] [x] [x] `exclude_outside` (type: AttrType.INT)
  - [x] [x] [x] `extrapolation_value` (type: AttrType.FLOAT)
  - [x] [x] [x] `mode` (type: AttrType.STRING)
  - [x] [x] [x] `nearest_mode` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `roi`
  - [x] [x] [x] `scales`
  - [x] [x] [x] `sizes`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `T2` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `coordinate_transformation_mode` (type: AttrType.STRING)
  - [x] [x] [x] `cubic_coeff_a` (type: AttrType.FLOAT)
  - [x] [x] [x] `exclude_outside` (type: AttrType.INT)
  - [x] [x] [x] `extrapolation_value` (type: AttrType.FLOAT)
  - [x] [x] [x] `mode` (type: AttrType.STRING)
  - [x] [x] [x] `nearest_mode` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `roi`
  - [x] [x] [x] `scales`
  - [x] [x] [x] `sizes`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `T2` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 18
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `antialias` (type: AttrType.INT)
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `coordinate_transformation_mode` (type: AttrType.STRING)
  - [x] [x] [x] `cubic_coeff_a` (type: AttrType.FLOAT)
  - [x] [x] [x] `exclude_outside` (type: AttrType.INT)
  - [x] [x] [x] `extrapolation_value` (type: AttrType.FLOAT)
  - [x] [x] [x] `keep_aspect_ratio_policy` (type: AttrType.STRING)
  - [x] [x] [x] `mode` (type: AttrType.STRING)
  - [x] [x] [x] `nearest_mode` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `roi`
  - [x] [x] [x] `scales`
  - [x] [x] [x] `sizes`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `T2` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 19
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `antialias` (type: AttrType.INT)
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `coordinate_transformation_mode` (type: AttrType.STRING)
  - [x] [x] [x] `cubic_coeff_a` (type: AttrType.FLOAT)
  - [x] [x] [x] `exclude_outside` (type: AttrType.INT)
  - [x] [x] [x] `extrapolation_value` (type: AttrType.FLOAT)
  - [x] [x] [x] `keep_aspect_ratio_policy` (type: AttrType.STRING)
  - [x] [x] [x] `mode` (type: AttrType.STRING)
  - [x] [x] [x] `nearest_mode` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `roi`
  - [x] [x] [x] `scales`
  - [x] [x] [x] `sizes`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `T2` supports: tensor(float16), tensor(float), tensor(double)

#### `ReverseSequence`
##### Opset 10
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `batch_axis` (type: AttrType.INT)
  - [x] [x] [x] `time_axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `sequence_lens`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

#### `RoiAlign`
##### Opset 10
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `mode` (type: AttrType.STRING)
  - [x] [x] [x] `output_height` (type: AttrType.INT)
  - [x] [x] [x] `output_width` (type: AttrType.INT)
  - [x] [x] [x] `sampling_ratio` (type: AttrType.INT)
  - [x] [x] [x] `spatial_scale` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `rois`
  - [x] [x] [x] `batch_indices`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T2` supports: tensor(int64)

##### Opset 16
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `coordinate_transformation_mode` (type: AttrType.STRING)
  - [x] [x] [x] `mode` (type: AttrType.STRING)
  - [x] [x] [x] `output_height` (type: AttrType.INT)
  - [x] [x] [x] `output_width` (type: AttrType.INT)
  - [x] [x] [x] `sampling_ratio` (type: AttrType.INT)
  - [x] [x] [x] `spatial_scale` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `rois`
  - [x] [x] [x] `batch_indices`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T2` supports: tensor(int64)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `coordinate_transformation_mode` (type: AttrType.STRING)
  - [x] [x] [x] `mode` (type: AttrType.STRING)
  - [x] [x] [x] `output_height` (type: AttrType.INT)
  - [x] [x] [x] `output_width` (type: AttrType.INT)
  - [x] [x] [x] `sampling_ratio` (type: AttrType.INT)
  - [x] [x] [x] `spatial_scale` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `rois`
  - [x] [x] [x] `batch_indices`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T2` supports: tensor(int64)

#### `RotaryEmbedding`
##### Opset 23
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `interleaved` (type: AttrType.INT)
  - [x] [x] [x] `num_heads` (type: AttrType.INT)
  - [x] [x] [x] `rotary_embedding_dim` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `cos_cache`
  - [x] [x] [x] `sin_cache`
  - [x] [x] [x] `position_ids`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float), tensor(float16), tensor(bfloat16)
  - [x] [x] [x] `M` supports: tensor(int64)

#### `Round`
##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `STFT`
##### Opset 17
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `onesided` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `signal`
  - [x] [x] [x] `frame_step`
  - [x] [x] [x] `window`
  - [x] [x] [x] `frame_length`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float), tensor(float16), tensor(double), tensor(bfloat16)
  - [x] [x] [x] `T2` supports: tensor(int32), tensor(int64)

#### `Scan`
##### Opset 8
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `body` (type: AttrType.GRAPH)
  - [x] [x] [x] `directions` (type: AttrType.INTS)
  - [x] [x] [x] `num_scan_inputs` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `sequence_lens`
  - [x] [x] [x] `initial_state_and_scan_inputs`
- **Outputs:**
  - [x] [x] [x] `final_state_and_scan_outputs`
- **Type Constraints:**
  - [x] [x] [x] `I` supports: tensor(int64)
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `body` (type: AttrType.GRAPH)
  - [x] [x] [x] `num_scan_inputs` (type: AttrType.INT)
  - [x] [x] [x] `scan_input_axes` (type: AttrType.INTS)
  - [x] [x] [x] `scan_input_directions` (type: AttrType.INTS)
  - [x] [x] [x] `scan_output_axes` (type: AttrType.INTS)
  - [x] [x] [x] `scan_output_directions` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `initial_state_and_scan_inputs`
- **Outputs:**
  - [x] [x] [x] `final_state_and_scan_outputs`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `body` (type: AttrType.GRAPH)
  - [x] [x] [x] `num_scan_inputs` (type: AttrType.INT)
  - [x] [x] [x] `scan_input_axes` (type: AttrType.INTS)
  - [x] [x] [x] `scan_input_directions` (type: AttrType.INTS)
  - [x] [x] [x] `scan_output_axes` (type: AttrType.INTS)
  - [x] [x] [x] `scan_output_directions` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `initial_state_and_scan_inputs`
- **Outputs:**
  - [x] [x] [x] `final_state_and_scan_outputs`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 16
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `body` (type: AttrType.GRAPH)
  - [x] [x] [x] `num_scan_inputs` (type: AttrType.INT)
  - [x] [x] [x] `scan_input_axes` (type: AttrType.INTS)
  - [x] [x] [x] `scan_input_directions` (type: AttrType.INTS)
  - [x] [x] [x] `scan_output_axes` (type: AttrType.INTS)
  - [x] [x] [x] `scan_output_directions` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `initial_state_and_scan_inputs`
- **Outputs:**
  - [x] [x] [x] `final_state_and_scan_outputs`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 19
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `body` (type: AttrType.GRAPH)
  - [x] [x] [x] `num_scan_inputs` (type: AttrType.INT)
  - [x] [x] [x] `scan_input_axes` (type: AttrType.INTS)
  - [x] [x] [x] `scan_input_directions` (type: AttrType.INTS)
  - [x] [x] [x] `scan_output_axes` (type: AttrType.INTS)
  - [x] [x] [x] `scan_output_directions` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `initial_state_and_scan_inputs`
- **Outputs:**
  - [x] [x] [x] `final_state_and_scan_outputs`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)

##### Opset 21
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `body` (type: AttrType.GRAPH)
  - [x] [x] [x] `num_scan_inputs` (type: AttrType.INT)
  - [x] [x] [x] `scan_input_axes` (type: AttrType.INTS)
  - [x] [x] [x] `scan_input_directions` (type: AttrType.INTS)
  - [x] [x] [x] `scan_output_axes` (type: AttrType.INTS)
  - [x] [x] [x] `scan_output_directions` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `initial_state_and_scan_inputs`
- **Outputs:**
  - [x] [x] [x] `final_state_and_scan_outputs`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4)

##### Opset 23
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `body` (type: AttrType.GRAPH)
  - [x] [x] [x] `num_scan_inputs` (type: AttrType.INT)
  - [x] [x] [x] `scan_input_axes` (type: AttrType.INTS)
  - [x] [x] [x] `scan_input_directions` (type: AttrType.INTS)
  - [x] [x] [x] `scan_output_axes` (type: AttrType.INTS)
  - [x] [x] [x] `scan_output_directions` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `initial_state_and_scan_inputs`
- **Outputs:**
  - [x] [x] [x] `final_state_and_scan_outputs`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1)

##### Opset 24
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `body` (type: AttrType.GRAPH)
  - [x] [x] [x] `num_scan_inputs` (type: AttrType.INT)
  - [x] [x] [x] `scan_input_axes` (type: AttrType.INTS)
  - [x] [x] [x] `scan_input_directions` (type: AttrType.INTS)
  - [x] [x] [x] `scan_output_axes` (type: AttrType.INTS)
  - [x] [x] [x] `scan_output_directions` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `initial_state_and_scan_inputs`
- **Outputs:**
  - [x] [x] [x] `final_state_and_scan_outputs`
- **Type Constraints:**
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1), tensor(float8e8m0)

#### `Scatter`
##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `indices`
  - [x] [x] [x] `updates`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `Tind` supports: tensor(int32), tensor(int64)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `indices`
  - [x] [x] [x] `updates`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `Tind` supports: tensor(int32), tensor(int64)

#### `ScatterElements`
##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `indices`
  - [x] [x] [x] `updates`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `Tind` supports: tensor(int32), tensor(int64)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `indices`
  - [x] [x] [x] `updates`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `Tind` supports: tensor(int32), tensor(int64)

##### Opset 16
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `reduction` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `indices`
  - [x] [x] [x] `updates`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `Tind` supports: tensor(int32), tensor(int64)

##### Opset 18
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `reduction` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `indices`
  - [x] [x] [x] `updates`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `Tind` supports: tensor(int32), tensor(int64)

#### `ScatterND`
##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `indices`
  - [x] [x] [x] `updates`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `indices`
  - [x] [x] [x] `updates`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 16
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `reduction` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `indices`
  - [x] [x] [x] `updates`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 18
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `reduction` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `indices`
  - [x] [x] [x] `updates`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

#### `Selu`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `alpha` (type: AttrType.FLOAT)
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
  - [x] [x] [x] `gamma` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `alpha` (type: AttrType.FLOAT)
  - [x] [x] [x] `gamma` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `alpha` (type: AttrType.FLOAT)
  - [x] [x] [x] `gamma` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `SequenceAt`
##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input_sequence`
  - [x] [x] [x] `position`
- **Outputs:**
  - [x] [x] [x] `tensor`
- **Type Constraints:**
  - [x] [x] [x] `S` supports: seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128))
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `I` supports: tensor(int32), tensor(int64)

#### `SequenceConstruct`
##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `inputs`
- **Outputs:**
  - [x] [x] [x] `output_sequence`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `S` supports: seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128))

#### `SequenceEmpty`
##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `dtype` (type: AttrType.INT)
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `S` supports: seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128))

#### `SequenceErase`
##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input_sequence`
  - [x] [x] [x] `position`
- **Outputs:**
  - [x] [x] [x] `output_sequence`
- **Type Constraints:**
  - [x] [x] [x] `S` supports: seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128))
  - [x] [x] [x] `I` supports: tensor(int32), tensor(int64)

#### `SequenceInsert`
##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input_sequence`
  - [x] [x] [x] `tensor`
  - [x] [x] [x] `position`
- **Outputs:**
  - [x] [x] [x] `output_sequence`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `S` supports: seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128))
  - [x] [x] [x] `I` supports: tensor(int32), tensor(int64)

#### `SequenceLength`
##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input_sequence`
- **Outputs:**
  - [x] [x] [x] `length`
- **Type Constraints:**
  - [x] [x] [x] `S` supports: seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128))
  - [x] [x] [x] `I` supports: tensor(int64)

#### `SequenceMap`
##### Opset 17
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `body` (type: AttrType.GRAPH)
- **Inputs:**
  - [x] [x] [x] `input_sequence`
  - [x] [x] [x] `additional_inputs`
- **Outputs:**
  - [x] [x] [x] `out_sequence`
- **Type Constraints:**
  - [x] [x] [x] `S` supports: seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128))
  - [x] [x] [x] `V` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128))

#### `Shape`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `shape`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `T1` supports: tensor(int64)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `shape`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `T1` supports: tensor(int64)

##### Opset 15
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `end` (type: AttrType.INT)
  - [x] [x] [x] `start` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `shape`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `T1` supports: tensor(int64)

##### Opset 19
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `end` (type: AttrType.INT)
  - [x] [x] [x] `start` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `shape`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)
  - [x] [x] [x] `T1` supports: tensor(int64)

##### Opset 21
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `end` (type: AttrType.INT)
  - [x] [x] [x] `start` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `shape`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4)
  - [x] [x] [x] `T1` supports: tensor(int64)

##### Opset 23
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `end` (type: AttrType.INT)
  - [x] [x] [x] `start` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `shape`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1)
  - [x] [x] [x] `T1` supports: tensor(int64)

##### Opset 24
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `end` (type: AttrType.INT)
  - [x] [x] [x] `start` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `shape`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1), tensor(float8e8m0)
  - [x] [x] [x] `T1` supports: tensor(int64)

#### `Shrink`
##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `bias` (type: AttrType.FLOAT)
  - [x] [x] [x] `lambd` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

#### `Sigmoid`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `Sign`
##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `Sin`
##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `Sinh`
##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `Size`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `size`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `T1` supports: tensor(int64)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `size`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `T1` supports: tensor(int64)

##### Opset 19
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `size`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)
  - [x] [x] [x] `T1` supports: tensor(int64)

##### Opset 21
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `size`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4)
  - [x] [x] [x] `T1` supports: tensor(int64)

##### Opset 23
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `size`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1)
  - [x] [x] [x] `T1` supports: tensor(int64)

##### Opset 24
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `size`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1), tensor(float8e8m0)
  - [x] [x] [x] `T1` supports: tensor(int64)

#### `Slice`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
  - [x] [x] [x] `ends` (type: AttrType.INTS)
  - [x] [x] [x] `starts` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 10
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `starts`
  - [x] [x] [x] `ends`
  - [x] [x] [x] `axes`
  - [x] [x] [x] `steps`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `Tind` supports: tensor(int32), tensor(int64)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `starts`
  - [x] [x] [x] `ends`
  - [x] [x] [x] `axes`
  - [x] [x] [x] `steps`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `Tind` supports: tensor(int32), tensor(int64)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `starts`
  - [x] [x] [x] `ends`
  - [x] [x] [x] `axes`
  - [x] [x] [x] `steps`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `Tind` supports: tensor(int32), tensor(int64)

#### `Softmax`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `SoftmaxCrossEntropyLoss`
##### Opset 12
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `ignore_index` (type: AttrType.INT)
  - [x] [x] [x] `reduction` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `scores`
  - [x] [x] [x] `labels`
  - [x] [x] [x] `weights`
- **Outputs:**
  - [x] [x] [x] `output`
  - [x] [x] [x] `log_prob`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `Tind` supports: tensor(int32), tensor(int64)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `ignore_index` (type: AttrType.INT)
  - [x] [x] [x] `reduction` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `scores`
  - [x] [x] [x] `labels`
  - [x] [x] [x] `weights`
- **Outputs:**
  - [x] [x] [x] `output`
  - [x] [x] [x] `log_prob`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
  - [x] [x] [x] `Tind` supports: tensor(int32), tensor(int64)

#### `Softplus`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `Softsign`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `SpaceToDepth`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `blocksize` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `blocksize` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

#### `Split`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `split` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `split`
- **Outputs:**
  - [x] [x] [x] `outputs...`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 2
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `split` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `outputs`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `split` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `outputs`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `split`
- **Outputs:**
  - [x] [x] [x] `outputs`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 18
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `num_outputs` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `split`
- **Outputs:**
  - [x] [x] [x] `outputs`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

#### `SplitToSequence`
##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `split`
- **Outputs:**
  - [x] [x] [x] `output_sequence`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `I` supports: tensor(int32), tensor(int64)
  - [x] [x] [x] `S` supports: seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128))

##### Opset 24
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `keepdims` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `split`
- **Outputs:**
  - [x] [x] [x] `output_sequence`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `I` supports: tensor(int32), tensor(int64)
  - [x] [x] [x] `S` supports: seq(tensor(uint8)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(int8)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(bfloat16)), seq(tensor(float16)), seq(tensor(float)), seq(tensor(double)), seq(tensor(string)), seq(tensor(bool)), seq(tensor(complex64)), seq(tensor(complex128))

#### `Sqrt`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `Squeeze`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `squeezed`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `squeezed`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `axes`
- **Outputs:**
  - [x] [x] [x] `squeezed`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 21
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `axes`
- **Outputs:**
  - [x] [x] [x] `squeezed`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4)

##### Opset 23
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `axes`
- **Outputs:**
  - [x] [x] [x] `squeezed`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1)

##### Opset 24
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `axes`
- **Outputs:**
  - [x] [x] [x] `squeezed`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1), tensor(float8e8m0)

#### `StringConcat`
##### Opset 20
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `Y`
- **Outputs:**
  - [x] [x] [x] `Z`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(string)

#### `StringNormalizer`
##### Opset 10
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `case_change_action` (type: AttrType.STRING)
  - [x] [x] [x] `is_case_sensitive` (type: AttrType.INT)
  - [x] [x] [x] `locale` (type: AttrType.STRING)
  - [x] [x] [x] `stopwords` (type: AttrType.STRINGS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`

#### `StringSplit`
##### Opset 20
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `delimiter` (type: AttrType.STRING)
  - [x] [x] [x] `maxsplit` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `Z`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(string)
  - [x] [x] [x] `T2` supports: tensor(string)
  - [x] [x] [x] `T3` supports: tensor(int64)

#### `Sub`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `broadcast` (type: AttrType.INT)
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `broadcast` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

##### Opset 14
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `Sum`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `data_0`
- **Outputs:**
  - [x] [x] [x] `sum`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data_0`
- **Outputs:**
  - [x] [x] [x] `sum`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 8
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data_0`
- **Outputs:**
  - [x] [x] [x] `sum`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data_0`
- **Outputs:**
  - [x] [x] [x] `sum`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)

#### `Swish`
##### Opset 24
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `alpha` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(bfloat16), tensor(double)

#### `Tan`
##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `Tanh`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `consumed_inputs` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `TensorScatter`
##### Opset 24
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `mode` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `past_cache`
  - [x] [x] [x] `update`
  - [x] [x] [x] `write_indices`
- **Outputs:**
  - [x] [x] [x] `present_cache`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1), tensor(float8e8m0)

#### `TfIdfVectorizer`
##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `max_gram_length` (type: AttrType.INT)
  - [x] [x] [x] `max_skip_count` (type: AttrType.INT)
  - [x] [x] [x] `min_gram_length` (type: AttrType.INT)
  - [x] [x] [x] `mode` (type: AttrType.STRING)
  - [x] [x] [x] `ngram_counts` (type: AttrType.INTS)
  - [x] [x] [x] `ngram_indexes` (type: AttrType.INTS)
  - [x] [x] [x] `pool_int64s` (type: AttrType.INTS)
  - [x] [x] [x] `pool_strings` (type: AttrType.STRINGS)
  - [x] [x] [x] `weights` (type: AttrType.FLOATS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(string), tensor(int32), tensor(int64)
  - [x] [x] [x] `T1` supports: tensor(float)

#### `ThresholdedRelu`
##### Opset 10
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `alpha` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)

##### Opset 22
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `alpha` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bfloat16), tensor(float16), tensor(float), tensor(double)

#### `Tile`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `tiles`
  - [x] [x] [x] `axis`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `T1` supports: tensor(int64)

##### Opset 6
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `repeats`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `T1` supports: tensor(int64)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `repeats`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `T1` supports: tensor(int64)

#### `TopK`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `k` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Values`
  - [x] [x] [x] `Indices`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `I` supports: tensor(int64)

##### Opset 10
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `K`
- **Outputs:**
  - [x] [x] [x] `Values`
  - [x] [x] [x] `Indices`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `I` supports: tensor(int64)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `largest` (type: AttrType.INT)
  - [x] [x] [x] `sorted` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `K`
- **Outputs:**
  - [x] [x] [x] `Values`
  - [x] [x] [x] `Indices`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)
  - [x] [x] [x] `I` supports: tensor(int64)

##### Opset 24
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `largest` (type: AttrType.INT)
  - [x] [x] [x] `sorted` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `K`
- **Outputs:**
  - [x] [x] [x] `Values`
  - [x] [x] [x] `Indices`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
  - [x] [x] [x] `I` supports: tensor(int64)

#### `Transpose`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `perm` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `transposed`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `perm` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `transposed`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 21
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `perm` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `transposed`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4)

##### Opset 23
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `perm` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `transposed`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1)

##### Opset 24
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `perm` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `transposed`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1), tensor(float8e8m0)

#### `Trilu`
##### Opset 14
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `upper` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `input`
  - [x] [x] [x] `k`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

#### `Unique`
##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `sorted` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `indices`
  - [x] [x] [x] `inverse_indices`
  - [x] [x] [x] `counts`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

#### `Unsqueeze`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `expanded`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 11
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axes` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `data`
- **Outputs:**
  - [x] [x] [x] `expanded`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 13
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `axes`
- **Outputs:**
  - [x] [x] [x] `expanded`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 21
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `axes`
- **Outputs:**
  - [x] [x] [x] `expanded`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4)

##### Opset 23
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `axes`
- **Outputs:**
  - [x] [x] [x] `expanded`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1)

##### Opset 24
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `data`
  - [x] [x] [x] `axes`
- **Outputs:**
  - [x] [x] [x] `expanded`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(uint4), tensor(int4), tensor(float4e2m1), tensor(float8e8m0)

#### `Upsample`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `height_scale` (type: AttrType.FLOAT)
  - [x] [x] [x] `mode` (type: AttrType.STRING)
  - [x] [x] [x] `width_scale` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bool), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)

##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `mode` (type: AttrType.STRING)
  - [x] [x] [x] `scales` (type: AttrType.FLOATS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `mode` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `scales`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 10
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `mode` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `scales`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

#### `Where`
##### Opset 9
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `condition`
  - [x] [x] [x] `X`
  - [x] [x] [x] `Y`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `B` supports: tensor(bool)
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

##### Opset 16
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `condition`
  - [x] [x] [x] `X`
  - [x] [x] [x] `Y`
- **Outputs:**
  - [x] [x] [x] `output`
- **Type Constraints:**
  - [x] [x] [x] `B` supports: tensor(bool)
  - [x] [x] [x] `T` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)

#### `Xor`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `axis` (type: AttrType.INT)
  - [x] [x] [x] `broadcast` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bool)
  - [x] [x] [x] `T1` supports: tensor(bool)

##### Opset 7
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `A`
  - [x] [x] [x] `B`
- **Outputs:**
  - [x] [x] [x] `C`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(bool)
  - [x] [x] [x] `T1` supports: tensor(bool)


### Domain: `ai.onnx.ml`
#### `ArrayFeatureExtractor`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Inputs:**
  - [x] [x] [x] `X`
  - [x] [x] [x] `Y`
- **Outputs:**
  - [x] [x] [x] `Z`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float), tensor(double), tensor(int64), tensor(int32), tensor(string)

#### `Binarizer`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `threshold` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float), tensor(double), tensor(int64), tensor(int32)

#### `CastMap`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `cast_to` (type: AttrType.STRING)
  - [x] [x] [x] `map_form` (type: AttrType.STRING)
  - [x] [x] [x] `max_map` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: map(int64, string), map(int64, float)
  - [x] [x] [x] `T2` supports: tensor(string), tensor(float), tensor(int64)

#### `CategoryMapper`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `cats_int64s` (type: AttrType.INTS)
  - [x] [x] [x] `cats_strings` (type: AttrType.STRINGS)
  - [x] [x] [x] `default_int64` (type: AttrType.INT)
  - [x] [x] [x] `default_string` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(string), tensor(int64)
  - [x] [x] [x] `T2` supports: tensor(string), tensor(int64)

#### `DictVectorizer`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `int64_vocabulary` (type: AttrType.INTS)
  - [x] [x] [x] `string_vocabulary` (type: AttrType.STRINGS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: map(string, int64), map(int64, string), map(int64, float), map(int64, double), map(string, float), map(string, double)
  - [x] [x] [x] `T2` supports: tensor(int64), tensor(float), tensor(double), tensor(string)

#### `FeatureVectorizer`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `inputdimensions` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(int32), tensor(int64), tensor(float), tensor(double)

#### `Imputer`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `imputed_value_floats` (type: AttrType.FLOATS)
  - [x] [x] [x] `imputed_value_int64s` (type: AttrType.INTS)
  - [x] [x] [x] `replaced_value_float` (type: AttrType.FLOAT)
  - [x] [x] [x] `replaced_value_int64` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float), tensor(double), tensor(int64), tensor(int32)

#### `LabelEncoder`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `classes_strings` (type: AttrType.STRINGS)
  - [x] [x] [x] `default_int64` (type: AttrType.INT)
  - [x] [x] [x] `default_string` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(string), tensor(int64)
  - [x] [x] [x] `T2` supports: tensor(string), tensor(int64)

##### Opset 2
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `default_float` (type: AttrType.FLOAT)
  - [x] [x] [x] `default_int64` (type: AttrType.INT)
  - [x] [x] [x] `default_string` (type: AttrType.STRING)
  - [x] [x] [x] `keys_floats` (type: AttrType.FLOATS)
  - [x] [x] [x] `keys_int64s` (type: AttrType.INTS)
  - [x] [x] [x] `keys_strings` (type: AttrType.STRINGS)
  - [x] [x] [x] `values_floats` (type: AttrType.FLOATS)
  - [x] [x] [x] `values_int64s` (type: AttrType.INTS)
  - [x] [x] [x] `values_strings` (type: AttrType.STRINGS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(string), tensor(int64), tensor(float)
  - [x] [x] [x] `T2` supports: tensor(string), tensor(int64), tensor(float)

##### Opset 4
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `default_float` (type: AttrType.FLOAT)
  - [x] [x] [x] `default_int64` (type: AttrType.INT)
  - [x] [x] [x] `default_string` (type: AttrType.STRING)
  - [x] [x] [x] `default_tensor` (type: AttrType.TENSOR)
  - [x] [x] [x] `keys_floats` (type: AttrType.FLOATS)
  - [x] [x] [x] `keys_int64s` (type: AttrType.INTS)
  - [x] [x] [x] `keys_strings` (type: AttrType.STRINGS)
  - [x] [x] [x] `keys_tensor` (type: AttrType.TENSOR)
  - [x] [x] [x] `values_floats` (type: AttrType.FLOATS)
  - [x] [x] [x] `values_int64s` (type: AttrType.INTS)
  - [x] [x] [x] `values_strings` (type: AttrType.STRINGS)
  - [x] [x] [x] `values_tensor` (type: AttrType.TENSOR)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(string), tensor(int64), tensor(float), tensor(int32), tensor(int16), tensor(double)
  - [x] [x] [x] `T2` supports: tensor(string), tensor(int64), tensor(float), tensor(int32), tensor(int16), tensor(double)

#### `LinearClassifier`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `classlabels_ints` (type: AttrType.INTS)
  - [x] [x] [x] `classlabels_strings` (type: AttrType.STRINGS)
  - [x] [x] [x] `coefficients` (type: AttrType.FLOATS)
  - [x] [x] [x] `intercepts` (type: AttrType.FLOATS)
  - [x] [x] [x] `multi_class` (type: AttrType.INT)
  - [x] [x] [x] `post_transform` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `Z`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float), tensor(double), tensor(int64), tensor(int32)
  - [x] [x] [x] `T2` supports: tensor(string), tensor(int64)

#### `LinearRegressor`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `coefficients` (type: AttrType.FLOATS)
  - [x] [x] [x] `intercepts` (type: AttrType.FLOATS)
  - [x] [x] [x] `post_transform` (type: AttrType.STRING)
  - [x] [x] [x] `targets` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float), tensor(double), tensor(int64), tensor(int32)

#### `Normalizer`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `norm` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float), tensor(double), tensor(int64), tensor(int32)

#### `OneHotEncoder`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `cats_int64s` (type: AttrType.INTS)
  - [x] [x] [x] `cats_strings` (type: AttrType.STRINGS)
  - [x] [x] [x] `zeros` (type: AttrType.INT)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(string), tensor(int64), tensor(int32), tensor(float), tensor(double)

#### `SVMClassifier`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `classlabels_ints` (type: AttrType.INTS)
  - [x] [x] [x] `classlabels_strings` (type: AttrType.STRINGS)
  - [x] [x] [x] `coefficients` (type: AttrType.FLOATS)
  - [x] [x] [x] `kernel_params` (type: AttrType.FLOATS)
  - [x] [x] [x] `kernel_type` (type: AttrType.STRING)
  - [x] [x] [x] `post_transform` (type: AttrType.STRING)
  - [x] [x] [x] `prob_a` (type: AttrType.FLOATS)
  - [x] [x] [x] `prob_b` (type: AttrType.FLOATS)
  - [x] [x] [x] `rho` (type: AttrType.FLOATS)
  - [x] [x] [x] `support_vectors` (type: AttrType.FLOATS)
  - [x] [x] [x] `vectors_per_class` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `Z`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float), tensor(double), tensor(int64), tensor(int32)
  - [x] [x] [x] `T2` supports: tensor(string), tensor(int64)

#### `SVMRegressor`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `coefficients` (type: AttrType.FLOATS)
  - [x] [x] [x] `kernel_params` (type: AttrType.FLOATS)
  - [x] [x] [x] `kernel_type` (type: AttrType.STRING)
  - [x] [x] [x] `n_supports` (type: AttrType.INT)
  - [x] [x] [x] `one_class` (type: AttrType.INT)
  - [x] [x] [x] `post_transform` (type: AttrType.STRING)
  - [x] [x] [x] `rho` (type: AttrType.FLOATS)
  - [x] [x] [x] `support_vectors` (type: AttrType.FLOATS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float), tensor(double), tensor(int64), tensor(int32)

#### `Scaler`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `offset` (type: AttrType.FLOATS)
  - [x] [x] [x] `scale` (type: AttrType.FLOATS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float), tensor(double), tensor(int64), tensor(int32)

#### `TreeEnsemble`
##### Opset 5
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `aggregate_function` (type: AttrType.INT)
  - [x] [x] [x] `leaf_targetids` (type: AttrType.INTS)
  - [x] [x] [x] `leaf_weights` (type: AttrType.TENSOR)
  - [x] [x] [x] `membership_values` (type: AttrType.TENSOR)
  - [x] [x] [x] `n_targets` (type: AttrType.INT)
  - [x] [x] [x] `nodes_falseleafs` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_falsenodeids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_featureids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_hitrates` (type: AttrType.TENSOR)
  - [x] [x] [x] `nodes_missing_value_tracks_true` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_modes` (type: AttrType.TENSOR)
  - [x] [x] [x] `nodes_splits` (type: AttrType.TENSOR)
  - [x] [x] [x] `nodes_trueleafs` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_truenodeids` (type: AttrType.INTS)
  - [x] [x] [x] `post_transform` (type: AttrType.INT)
  - [x] [x] [x] `tree_roots` (type: AttrType.INTS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float), tensor(double), tensor(float16)

#### `TreeEnsembleClassifier`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `base_values` (type: AttrType.FLOATS)
  - [x] [x] [x] `class_ids` (type: AttrType.INTS)
  - [x] [x] [x] `class_nodeids` (type: AttrType.INTS)
  - [x] [x] [x] `class_treeids` (type: AttrType.INTS)
  - [x] [x] [x] `class_weights` (type: AttrType.FLOATS)
  - [x] [x] [x] `classlabels_int64s` (type: AttrType.INTS)
  - [x] [x] [x] `classlabels_strings` (type: AttrType.STRINGS)
  - [x] [x] [x] `nodes_falsenodeids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_featureids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_hitrates` (type: AttrType.FLOATS)
  - [x] [x] [x] `nodes_missing_value_tracks_true` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_modes` (type: AttrType.STRINGS)
  - [x] [x] [x] `nodes_nodeids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_treeids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_truenodeids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_values` (type: AttrType.FLOATS)
  - [x] [x] [x] `post_transform` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `Z`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float), tensor(double), tensor(int64), tensor(int32)
  - [x] [x] [x] `T2` supports: tensor(string), tensor(int64)

##### Opset 3
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `base_values` (type: AttrType.FLOATS)
  - [x] [x] [x] `base_values_as_tensor` (type: AttrType.TENSOR)
  - [x] [x] [x] `class_ids` (type: AttrType.INTS)
  - [x] [x] [x] `class_nodeids` (type: AttrType.INTS)
  - [x] [x] [x] `class_treeids` (type: AttrType.INTS)
  - [x] [x] [x] `class_weights` (type: AttrType.FLOATS)
  - [x] [x] [x] `class_weights_as_tensor` (type: AttrType.TENSOR)
  - [x] [x] [x] `classlabels_int64s` (type: AttrType.INTS)
  - [x] [x] [x] `classlabels_strings` (type: AttrType.STRINGS)
  - [x] [x] [x] `nodes_falsenodeids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_featureids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_hitrates` (type: AttrType.FLOATS)
  - [x] [x] [x] `nodes_hitrates_as_tensor` (type: AttrType.TENSOR)
  - [x] [x] [x] `nodes_missing_value_tracks_true` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_modes` (type: AttrType.STRINGS)
  - [x] [x] [x] `nodes_nodeids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_treeids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_truenodeids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_values` (type: AttrType.FLOATS)
  - [x] [x] [x] `nodes_values_as_tensor` (type: AttrType.TENSOR)
  - [x] [x] [x] `post_transform` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `Z`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float), tensor(double), tensor(int64), tensor(int32)
  - [x] [x] [x] `T2` supports: tensor(string), tensor(int64)

##### Opset 5
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `base_values` (type: AttrType.FLOATS)
  - [x] [x] [x] `base_values_as_tensor` (type: AttrType.TENSOR)
  - [x] [x] [x] `class_ids` (type: AttrType.INTS)
  - [x] [x] [x] `class_nodeids` (type: AttrType.INTS)
  - [x] [x] [x] `class_treeids` (type: AttrType.INTS)
  - [x] [x] [x] `class_weights` (type: AttrType.FLOATS)
  - [x] [x] [x] `class_weights_as_tensor` (type: AttrType.TENSOR)
  - [x] [x] [x] `classlabels_int64s` (type: AttrType.INTS)
  - [x] [x] [x] `classlabels_strings` (type: AttrType.STRINGS)
  - [x] [x] [x] `nodes_falsenodeids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_featureids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_hitrates` (type: AttrType.FLOATS)
  - [x] [x] [x] `nodes_hitrates_as_tensor` (type: AttrType.TENSOR)
  - [x] [x] [x] `nodes_missing_value_tracks_true` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_modes` (type: AttrType.STRINGS)
  - [x] [x] [x] `nodes_nodeids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_treeids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_truenodeids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_values` (type: AttrType.FLOATS)
  - [x] [x] [x] `nodes_values_as_tensor` (type: AttrType.TENSOR)
  - [x] [x] [x] `post_transform` (type: AttrType.STRING)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
  - [x] [x] [x] `Z`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float), tensor(double), tensor(int64), tensor(int32)
  - [x] [x] [x] `T2` supports: tensor(string), tensor(int64)

#### `TreeEnsembleRegressor`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `aggregate_function` (type: AttrType.STRING)
  - [x] [x] [x] `base_values` (type: AttrType.FLOATS)
  - [x] [x] [x] `n_targets` (type: AttrType.INT)
  - [x] [x] [x] `nodes_falsenodeids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_featureids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_hitrates` (type: AttrType.FLOATS)
  - [x] [x] [x] `nodes_missing_value_tracks_true` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_modes` (type: AttrType.STRINGS)
  - [x] [x] [x] `nodes_nodeids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_treeids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_truenodeids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_values` (type: AttrType.FLOATS)
  - [x] [x] [x] `post_transform` (type: AttrType.STRING)
  - [x] [x] [x] `target_ids` (type: AttrType.INTS)
  - [x] [x] [x] `target_nodeids` (type: AttrType.INTS)
  - [x] [x] [x] `target_treeids` (type: AttrType.INTS)
  - [x] [x] [x] `target_weights` (type: AttrType.FLOATS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float), tensor(double), tensor(int64), tensor(int32)

##### Opset 3
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `aggregate_function` (type: AttrType.STRING)
  - [x] [x] [x] `base_values` (type: AttrType.FLOATS)
  - [x] [x] [x] `base_values_as_tensor` (type: AttrType.TENSOR)
  - [x] [x] [x] `n_targets` (type: AttrType.INT)
  - [x] [x] [x] `nodes_falsenodeids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_featureids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_hitrates` (type: AttrType.FLOATS)
  - [x] [x] [x] `nodes_hitrates_as_tensor` (type: AttrType.TENSOR)
  - [x] [x] [x] `nodes_missing_value_tracks_true` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_modes` (type: AttrType.STRINGS)
  - [x] [x] [x] `nodes_nodeids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_treeids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_truenodeids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_values` (type: AttrType.FLOATS)
  - [x] [x] [x] `nodes_values_as_tensor` (type: AttrType.TENSOR)
  - [x] [x] [x] `post_transform` (type: AttrType.STRING)
  - [x] [x] [x] `target_ids` (type: AttrType.INTS)
  - [x] [x] [x] `target_nodeids` (type: AttrType.INTS)
  - [x] [x] [x] `target_treeids` (type: AttrType.INTS)
  - [x] [x] [x] `target_weights` (type: AttrType.FLOATS)
  - [x] [x] [x] `target_weights_as_tensor` (type: AttrType.TENSOR)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float), tensor(double), tensor(int64), tensor(int32)

##### Opset 5
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `aggregate_function` (type: AttrType.STRING)
  - [x] [x] [x] `base_values` (type: AttrType.FLOATS)
  - [x] [x] [x] `base_values_as_tensor` (type: AttrType.TENSOR)
  - [x] [x] [x] `n_targets` (type: AttrType.INT)
  - [x] [x] [x] `nodes_falsenodeids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_featureids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_hitrates` (type: AttrType.FLOATS)
  - [x] [x] [x] `nodes_hitrates_as_tensor` (type: AttrType.TENSOR)
  - [x] [x] [x] `nodes_missing_value_tracks_true` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_modes` (type: AttrType.STRINGS)
  - [x] [x] [x] `nodes_nodeids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_treeids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_truenodeids` (type: AttrType.INTS)
  - [x] [x] [x] `nodes_values` (type: AttrType.FLOATS)
  - [x] [x] [x] `nodes_values_as_tensor` (type: AttrType.TENSOR)
  - [x] [x] [x] `post_transform` (type: AttrType.STRING)
  - [x] [x] [x] `target_ids` (type: AttrType.INTS)
  - [x] [x] [x] `target_nodeids` (type: AttrType.INTS)
  - [x] [x] [x] `target_treeids` (type: AttrType.INTS)
  - [x] [x] [x] `target_weights` (type: AttrType.FLOATS)
  - [x] [x] [x] `target_weights_as_tensor` (type: AttrType.TENSOR)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Y`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: tensor(float), tensor(double), tensor(int64), tensor(int32)

#### `ZipMap`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `classlabels_int64s` (type: AttrType.INTS)
  - [x] [x] [x] `classlabels_strings` (type: AttrType.STRINGS)
- **Inputs:**
  - [x] [x] [x] `X`
- **Outputs:**
  - [x] [x] [x] `Z`
- **Type Constraints:**
  - [x] [x] [x] `T` supports: seq(map(string, float)), seq(map(int64, float))


### Domain: `ai.onnx.preview.training`
#### `Adagrad`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `decay_factor` (type: AttrType.FLOAT)
  - [x] [x] [x] `epsilon` (type: AttrType.FLOAT)
  - [x] [x] [x] `norm_coefficient` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `R`
  - [x] [x] [x] `T`
  - [x] [x] [x] `inputs`
- **Outputs:**
  - [x] [x] [x] `outputs`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float), tensor(double)
  - [x] [x] [x] `T2` supports: tensor(int64)
  - [x] [x] [x] `T3` supports: tensor(float), tensor(double)

#### `Adam`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `alpha` (type: AttrType.FLOAT)
  - [x] [x] [x] `beta` (type: AttrType.FLOAT)
  - [x] [x] [x] `epsilon` (type: AttrType.FLOAT)
  - [x] [x] [x] `norm_coefficient` (type: AttrType.FLOAT)
  - [x] [x] [x] `norm_coefficient_post` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `R`
  - [x] [x] [x] `T`
  - [x] [x] [x] `inputs`
- **Outputs:**
  - [x] [x] [x] `outputs`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float), tensor(double)
  - [x] [x] [x] `T2` supports: tensor(int64)
  - [x] [x] [x] `T3` supports: tensor(float), tensor(double)

#### `Gradient`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `xs` (type: AttrType.STRINGS)
  - [x] [x] [x] `y` (type: AttrType.STRING)
  - [x] [x] [x] `zs` (type: AttrType.STRINGS)
- **Inputs:**
  - [x] [x] [x] `Inputs`
- **Outputs:**
  - [x] [x] [x] `Outputs`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
  - [x] [x] [x] `T2` supports: tensor(float16), tensor(float), tensor(double)

#### `Momentum`
##### Opset 1
- [x] [x] [x] Implementation
- [x] [x] [x] Shape Inference
- [x] [x] [x] Type Inference
- **Attributes:**
  - [x] [x] [x] `alpha` (type: AttrType.FLOAT)
  - [x] [x] [x] `beta` (type: AttrType.FLOAT)
  - [x] [x] [x] `mode` (type: AttrType.STRING)
  - [x] [x] [x] `norm_coefficient` (type: AttrType.FLOAT)
- **Inputs:**
  - [x] [x] [x] `R`
  - [x] [x] [x] `T`
  - [x] [x] [x] `inputs`
- **Outputs:**
  - [x] [x] [x] `outputs`
- **Type Constraints:**
  - [x] [x] [x] `T1` supports: tensor(float), tensor(double)
  - [x] [x] [x] `T2` supports: tensor(int64)
  - [x] [x] [x] `T3` supports: tensor(float), tensor(double)


## 3. Integration with ml-switcheroo
This section tracks the end-to-end integration of the onnx9000 engine within the parent `ml-switcheroo` project ecosystem.

### 3.1 IR to ONNX Export
- [x] [x] [x] Translate `ml-switcheroo` internal IR nodes to standard ONNX operators.
- [x] [x] [x] Correctly map internal data types to ONNX `TensorProto` data types.
- [x] [x] [x] Handle dynamic shapes and broadcast semantics during translation.
- [x] [x] [x] Serialize the compiled graph into a valid ONNX `ModelProto` binary.

### 3.2 In-Browser Training & Serving Pipeline
- [x] [x] [x] **Training in the Browser:**
  - [x] [x] [x] Implement/Integrate training graph builder compatible with `onnxruntime-web` training APIs.
  - [x] [x] [x] Support inserting loss functions and optimizers into the exported ONNX model.
  - [x] [x] [x] Manage WebAssembly (WASM) / WebGPU memory efficiently during training iterations.
- [x] [x] [x] **Serving in the Browser:**
  - [x] [x] [x] Seamlessly transition the trained model state (updated weights) to an inference session.
  - [x] [x] [x] Perform fast, zero-copy (where possible) browser-based inference.

### 3.3 External Pipeline Interoperability
- [x] [x] [x] **Download ONNX Feature:**
  - [x] [x] [x] Expose an API/UI button to download the strictly compliant `.onnx` model file.
- [x] [x] [x] **Official Pipeline Compatibility:**
  - [x] [x] [x] Verify the downloaded model runs correctly on standard `onnxruntime` (Python/C++).
  - [x] [x] [x] Verify the downloaded model can be used with official ONNX training servers (e.g., ORTModule/PyTorch).
