/* eslint-disable */
import { Program, Function, Block } from './mil/ast.js';
import { Model } from './schema.js';
import { parseModel } from './schema.js';
import { BufferReader } from '@onnx9000/core';

export class MLPackageLoader {
  // 216. Implement .mlmodel and .mlpackage loader/unzipper in JS
  static async loadFromZip(
    jszipInstance: ReturnType<typeof JSON.parse>,
    zipData: Uint8Array,
  ): Promise<{ model: Model; weights: Uint8Array }> {
    const zip = new jszipInstance();
    await zip.loadAsync(zipData);

    // Read model
    const modelFile = zip.file('Data/com.apple.CoreML/model.mlmodel');
    if (!modelFile) throw new Error('model.mlmodel not found in package');
    const modelBytes = await modelFile.async('uint8array');

    const reader = new BufferReader(modelBytes);
    const model = await parseModel(reader);

    // Read weights
    const weightFile = zip.file('Data/com.apple.CoreML/weights/weight.bin');
    const weights = weightFile ? await weightFile.async('uint8array') : new Uint8Array(0);

    return { model, weights };
  }

  // 217. Parse MILSpec.Program back into the TypeScript AST representation
  static parseMILProgram(model: Model): Program {
    const prog = new Program();

    // Mock parsing MILSpec to Program AST
    if (model.mlProgram) {
      // iterate over model.mlProgram.functions and reconstruct the `Function` / `Block` objects
      const fn = new Function('main', [], []);
      prog.addFunction(fn);
    }

    // 218. Parse Apple NeuralNetwork V1-V3 layers (legacy protobuf) into the AST
    if (model.neuralNetwork) {
      // iterate over model.neuralNetwork.layers and map to MIL Ops
    }

    return prog;
  }
}
