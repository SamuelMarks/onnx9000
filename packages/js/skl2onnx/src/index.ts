/**
 * @fileoverview Scikit-Learn estimator to `ai.onnx.ml` converter for `@onnx9000/skl2onnx`.
 * Translates classical ML estimators (`LogisticRegression`, `RandomForestClassifier`, `StandardScaler`, `SVC`) into ONNX ML operators.
 */

/**
 * Converts scikit-learn estimator specifications into `ai.onnx.ml` operator types.
 */
import { ConversionError } from '@onnx9000/core';
/** Represents the `SKL2ONNXConverter` class. */
export class SKL2ONNXConverter {
  /** Mapping from scikit-learn estimator names to `ai.onnx.ml` operator types. */
  private static readonly ESTIMATOR_MAP: Readonly<Record<string, string>> = {
    LogisticRegression: 'LinearClassifier',
    Ridge: 'LinearRegressor',
    LinearRegression: 'LinearRegressor',
    RandomForestClassifier: 'TreeEnsembleClassifier',
    RandomForestRegressor: 'TreeEnsembleRegressor',
    StandardScaler: 'Scaler',
    MinMaxScaler: 'Scaler',
    SVC: 'SVMClassifier',
    SVR: 'SVMRegressor',
  };

  /**
   * Maps a scikit-learn estimator class name to its `ai.onnx.ml` operator type.
   * @param estimatorName Scikit-learn class name.
   * @returns Corresponding `ai.onnx.ml` operator type string.
   */
  public mapEstimator(estimatorName: string): string {
    return SKL2ONNXConverter.ESTIMATOR_MAP[estimatorName] ?? estimatorName;
  }

  /**
   * Converts a scikit-learn model string into an ONNX IR summary.
   * @param sklModelString Scikit-learn model descriptor string.
   * @returns Formatted ONNX IR string.
   * @throws Error if `sklModelString` is empty.
   */
  public convert(sklModelString: string): string {
    if (!sklModelString) {
      throw new ConversionError('Invalid model string');
    }
    return `[ONNX-IR] from skl ${sklModelString}`;
  }
}
