import { Graph } from '@onnx9000/core';
import { ONNXToMILConverter } from './converter.js';
import { MILToONNXConverter } from './importer.js';
import { Program } from './mil/ast.js';
import { MLPackageBuilder } from './mlpackage.js';
import { Model } from './schema.js';

export function convertToCoreML(
  graph: Graph,
  options: { dynamicBatching?: boolean } = {},
): Program {
  const converter = new ONNXToMILConverter(graph, options);
  return converter.convert();
}

export function importCoreML(program: Program): Graph {
  const converter = new MILToONNXConverter(program);
  return converter.convert();
}

export function buildMLPackage(
  model: Model,
  weights: Uint8Array = new Uint8Array(0),
  options: {
    classLabels?: string[];
    outputMappings?: Record<string, string>;
    stateful?: boolean;
    generateSwiftBoilerplate?: boolean;
    computePrecision?: 'Float16' | 'Float32';
    imageInputs?: Record<
      string,
      { blueBias?: number; greenBias?: number; redBias?: number; imageScale?: number }
    >;
    classifierOutputs?: string[];
    visionFrameworkDescription?: string;
    sequenceInputs?: string[];
    vocabularyFiles?: Record<string, Uint8Array>;
  } = {},
): MLPackageBuilder {
  return new MLPackageBuilder(model, weights, options);
}
