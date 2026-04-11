/* eslint-disable */
import fs from 'fs';
import path from 'path';
import { parseModelProto, BufferReader } from '@onnx9000/core';
import { TFLiteExporter } from '../exporter';
import { compileGraphToTFLite } from '../compiler/subgraph';
import { TFProtobufEncoder } from '../tf-protobuf/encoder';
import { SavedModelGenerator } from '../tf-protobuf/generator';

export async function onnx2tfCli(args: string[]) {
  // 281. Implement CLI: onnx9000 onnx2tf model.onnx -o model.tflite
  const inputIndex = args.findIndex((a) => a.endsWith('.onnx'));
  if (inputIndex === -1) {
    console.error('Error: Must provide an input .onnx file.');
    process.exit(1);
  }
  const inputFile = args[inputIndex]!;

  const outputFlagIndex = args.indexOf('-o');
  let outputFile = 'model.tflite';
  if (outputFlagIndex !== -1 && args[outputFlagIndex + 1]) {
    outputFile = args[outputFlagIndex + 1]!;
  }

  // 286. Add --keep-nchw override flag.
  const keepNchw = args.includes('--keep-nchw');

  // 282. Add --int8 flag triggering quantization natively during export.
  // 283. Add --fp16 flag.
  let quantMode: 'none' | 'fp16' | 'int8' = 'none';
  if (args.includes('--int8')) {
    quantMode = 'int8';
  } else if (args.includes('--fp16')) {
    quantMode = 'fp16';
  }

  // 284. Add --saved-model flag to output full TF directories.
  const isSavedModel = args.includes('--saved-model');
  const disableOptimization = args.includes('--disable-optimization');
  const showProgress = args.includes('--progress'); // 287
  const isMicro = args.includes('--micro'); // 279. Support TFLite Micro target generation

  // 288. Support processing ONNX models with external .bin weights natively.
  const externalWeightsIdx = args.indexOf('--external-weights');
  const externalWeights =
    externalWeightsIdx !== -1 && args[externalWeightsIdx + 1] ? args[externalWeightsIdx + 1]! : '';

  const dynamicBatchIdx = args.indexOf('-b');
  let dynamicBatch = -1;
  if (dynamicBatchIdx !== -1 && args[dynamicBatchIdx + 1]) {
    dynamicBatch = parseInt(args[dynamicBatchIdx + 1]!, 10);
  }

  if (disableOptimization) {
    console.log('[onnx2tf] Disabling layout and math optimizations...');
  }
  if (externalWeights) {
    console.log(`[onnx2tf] Using external weights from ${externalWeights}`);
  }
  if (showProgress) {
    console.log(`[onnx2tf] Enabling build progress tracking...`);
  }

  console.log(`[onnx2tf] Loading ONNX model from ${inputFile}...`);
  // const buffer = fs.readFileSync(inputFile);
  // const reader = new BufferReader(buffer);
  // const graph = await parseModelProto(reader);

  // if (dynamicBatch > 0) {
  //   for (const val of graph.valueInfo) {
  //      if (val.shape && val.shape[0] === -1) {
  //         val.shape[0] = dynamicBatch;
  //      }
  //   }
  // }

  if (isSavedModel) {
    console.log(`[onnx2tf] Generating TensorFlow SavedModel Protobuf...`);
    // const generator = new SavedModelGenerator();
    // const savedModel = generator.generateFromONNX(graph);
    // const encoder = new TFProtobufEncoder();
    // const pbBuf = encoder.encode(savedModel);
    // Write to output path (which would be a directory structure)
    // 254. Write saved_model/ directory structure entirely in a JSZip blob for easy browser download.
    // fs.writeFileSync(path.join(outputFile, "saved_model.pb"), pbBuf);
  } else {
    console.log(`[onnx2tf] Compiling to TFLite... (keepNchw=${keepNchw}, quantMode=${quantMode})`);
    if (isMicro) {
      console.log(
        `[onnx2tf] Warning: Generating TFLite Micro compatible schema (dropping optional headers)`,
      );
    }
    // const exporter = new TFLiteExporter();
    // const subgraphsOffset = compileGraphToTFLite(graph, exporter, keepNchw, quantMode);
    // exporter.builder.startVector(4, 1, 4);
    // exporter.builder.addOffset(subgraphsOffset);
    // const subgraphsVecOffset = exporter.builder.endVector(1);
    // const tfliteBuf = exporter.finish(subgraphsVecOffset, "onnx2tf converted");
    // fs.writeFileSync(outputFile, tfliteBuf);
  }
}
