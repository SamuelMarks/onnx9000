/**
 * @fileoverview parser module for converters.
 */
/**
 * LibSVM parser.
 */

export interface LibSVMModel {
  /** Property `svmType`. */
  svmType: string;
  /** Property `kernelType`. */
  kernelType: string;
  /** Property `rho`. */
  rho: number;
  /** Property `coefs`. */
  coefs: number[];
}

/**
 * Executes `parseLibSVM`.
 * @param content Parameter `content`.
 * @returns Result of `parseLibSVM`.
 */
export function parseLibSVM(content: string): LibSVMModel {
  const lines = content.trim().split('\n');

  let svmType = 'c_svc';
  let kernelType = 'rbf';
  let rho = 0.0;
  const coefs: number[] = [];

  let svMode = false;
  for (let line of lines) {
    line = line.trim();
    if (!line) continue;

    if (svMode) {
      const parts = line.split(/\s+/);
      coefs.push(parseFloat(parts[0] as string) || 0);
    } else {
      if (line.startsWith('svm_type')) {
        svmType = line.split(/\s+/)[1] || 'c_svc';
      } else if (line.startsWith('kernel_type')) {
        kernelType = line.split(/\s+/)[1] || 'rbf';
      } else if (line.startsWith('rho')) {
        const rhoStr = line.split(/\s+/)[1];
        if (rhoStr) rho = parseFloat(rhoStr);
      } else if (line === 'SV') {
        svMode = true;
      }
    }
  }

  return { svmType, kernelType, rho, coefs };
}
