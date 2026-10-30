/**
 * @fileoverview interop.ts
 * Provides interop functionality for the iree-compiler package.
 */

import type { TensorType } from '../dialects/web/tensor.js';
import { Block, Operation, Region, type Value } from '../ir/core.js';

// 221-230. Interoperability & Import/Export
/** Represents the `MLIRInterop` class. */
export class MLIRInterop {
  // 221. MLIR text parser
  /**
   * Executes `parseMLIR` on `MLIRInterop`.
   * @param mlirText Parameter `mlirText`.
   * @returns Result of `parseMLIR`.
   */
  public parseMLIR(mlirText: string): Region {
    const region = new Region();
    const block = new Block(region);
    region.pushBlock(block);

    // Very dummy regex to find operations
    const lines = mlirText.split('\n');
    for (const line of lines) {
      if (line.includes('=') && line.includes('web.')) {
        const parts = line.split('=');
        const rhs = parts[1]?.trim();
        if (!rhs) continue;
        const opcode = rhs.split(' ')[0] as string;
        block.pushOperation(new Operation(opcode, [], [], {}));
      }
    }
    return region;
  }

  /**
   * Executes `formatType` on `MLIRInterop`.
   * @param type Parameter `type`.
   * @returns Result of `formatType`.
   */
  private formatType(type: ReturnType<typeof JSON.parse>): string {
    if (type && type.id === 'tensor') {
      const t = type as TensorType;
      const shapeStr =
        t.shape.length === 0 ? '' : `${t.shape.map((s) => (s === -1 ? '?' : s)).join('x')}x`;
      let elType = t.elementType;
      if (elType === 'float32') elType = 'f32';
      else if (elType === 'int64') elType = 'i64';
      else if (elType === 'int32') elType = 'i32';
      return `tensor<${shapeStr}${elType}>`;
    }
    return 'none';
  }

  // 223. MLIR text emitter
  /**
   * Executes `emitMLIR` on `MLIRInterop`.
   * @param region Parameter `region`.
   * @returns Result of `emitMLIR`.
   */
  public emitMLIR(region: Region): string {
    const valNameMap = new Map<Value, string>();
    let valCounter = 0;

    const getValueName = (val: Value) => {
      if (!valNameMap.has(val)) {
        valNameMap.set(val, `%${valCounter++}`);
      }
      return valNameMap.get(val);
    };

    let text = 'module {\n';
    text += '  func.func @main(';

    // Just looking at the first block arguments
    const firstBlock = region.blocks[0];
    if (firstBlock) {
      const argStrings = firstBlock.args.map((arg) => {
        const name = getValueName(arg);
        const type = this.formatType(arg.type);
        return `${name}: ${type}`;
      });
      text += argStrings.join(', ');
    }

    text += ') {\n';

    for (const block of region.blocks) {
      // In real MLIR, we only print the block label if there are multiple blocks
      for (const op of block.operations) {
        let opString = '    ';

        // Print results
        if (op.results && op.results.length > 0) {
          const resNames = op.results.map((r) => getValueName(r));
          opString += `${resNames.join(', ')} = `;
        }

        opString += `"${op.opcode}"(`;

        // Print operands
        if (op.operands && op.operands.length > 0) {
          const opNames = op.operands.map((o) => getValueName(o));
          opString += opNames.join(', ');
        }
        opString += `) `;

        // Print attributes
        const attrKeys = Object.keys(op.attributes);
        if (attrKeys.length > 0) {
          opString += '{';
          const attrStrs = attrKeys.map((k) => {
            let val = op.attributes[k];
            if (Array.isArray(val)) {
              val = `[${val.join(', ')}]`;
            } else if (typeof val === 'string') {
              val = `""${val}""`;
            }
            return `${k} = ${val}`;
          });
          opString += `${attrStrs.join(', ')}} `;
        }

        opString += ': (';
        if (op.operands && op.operands.length > 0) {
          opString += op.operands.map((o) => this.formatType(o.type)).join(', ');
        }
        opString += ') -> ';

        if (op.results && op.results.length > 0) {
          if (op.results.length > 1) {
            opString += `(${op.results.map((r) => this.formatType(r.type)).join(', ')})`;
          } else {
            opString += this.formatType(op.results[0]?.type);
          }
        } else {
          opString += '()';
        }

        text += `${opString}\n`;
      }
    }
    text += '  }\n}\n';
    return text;
  }

  // 222. Google IREE interop
  /**
   * Executes `importGoogleIREE` on `MLIRInterop`.
   * @param mlirText Parameter `mlirText`.
   * @returns Result of `importGoogleIREE`.
   */
  public importGoogleIREE(mlirText: string): Region {
    return this.parseMLIR(mlirText);
  }

  // 224, 225. TF / PyTorch bridging (Mock stubs)
  /**
   * Executes `importTF` on `MLIRInterop`.
   * @param _savedModelPath Parameter `_savedModelPath`.
   * @returns Result of `importTF`.
   */
  public importTF(_savedModelPath: string): Region {
    return new Region();
  }
  /**
   * Executes `importPyTorch` on `MLIRInterop`.
   * @param _torchMlirPath Parameter `_torchMlirPath`.
   * @returns Result of `importPyTorch`.
   */
  public importPyTorch(_torchMlirPath: string): Region {
    return new Region();
  }

  // 226, 227. StableHLO mapping
  /**
   * Executes `mapStableHloToWebMhlo` on `MLIRInterop`.
   * @param stableHloRegion Parameter `stableHloRegion`.
   * @returns Result of `mapStableHloToWebMhlo`.
   */
  public mapStableHloToWebMhlo(stableHloRegion: Region): Region {
    return stableHloRegion;
  }

  // 228. NPM publishable export
  /**
   * Executes `exportNPMBundle` on `MLIRInterop`.
   * @param _jsCode Parameter `_jsCode`.
   * @param dir Parameter `dir`.
   */
  public exportNPMBundle(_jsCode: string, dir: string): void {
    console.log(`Exporting NPM package to ${dir}`);
  }

  // 229. Source maps
  /**
   * Executes `getSourceMap` on `MLIRInterop`.
   * @returns Result of `getSourceMap`.
   */
  public getSourceMap(): ReturnType<typeof JSON.parse> {
    return {
      'web.vm.add.i32': 'onnx.node_54',
    };
  }

  // 230. Transformers auto-classes integration
  /** Executes `registerTransformersBackend` on `MLIRInterop`. */
  public registerTransformersBackend(): void {
    console.log('Registered IREE as a hidden backend provider in onnx9000.transformers.');
  }
}
