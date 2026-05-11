/**
 * LibSVM parser.
 */

export interface LibSVMModel {
  svmType: string;
  kernelType: string;
  rho: number;
  coefs: number[];
}

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
      if (parts.length > 0 && parts[0] !== undefined) {
        const coef = parseFloat(parts[0]);
        if (!isNaN(coef)) {
          coefs.push(coef);
        }
      }
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
