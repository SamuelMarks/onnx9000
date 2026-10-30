/**
 * @fileoverview scf.ts
 * Provides scf functionality for the iree-compiler package.
 */
import { Operation, type Region, type Type, type Value } from '../../ir/core.js';

// 92. web.scf dialect
/**
 * Executes `forOp`.
 * @param lowerBound Parameter `lowerBound`.
 * @param upperBound Parameter `upperBound`.
 * @param step Parameter `step`.
 * @param initArgs Parameter `initArgs`.
 * @param body Parameter `body`.
 * @returns Result of `forOp`.
 */
export function forOp(
  lowerBound: Value,
  upperBound: Value,
  step: Value,
  initArgs: Value[],
  body: Region,
): Operation {
  return new Operation(
    'web.scf.for',
    [lowerBound, upperBound, step, ...initArgs],
    initArgs.map((a) => a.type),
    {},
    [body],
  );
}

/**
 * Executes `yieldOp`.
 * @param results Parameter `results`.
 * @returns Result of `yieldOp`.
 */
export function yieldOp(results: Value[]): Operation {
  return new Operation('web.scf.yield', results, []);
}

/**
 * Executes `ifOp`.
 * @param condition Parameter `condition`.
 * @param trueRegion Parameter `trueRegion`.
 * @param falseRegion Parameter `falseRegion`.
 * @param resultTypes Parameter `resultTypes`.
 * @returns Result of `ifOp`.
 */
export function ifOp(
  condition: Value,
  trueRegion: Region,
  falseRegion: Region,
  resultTypes: Type[],
): Operation {
  return new Operation('web.scf.if', [condition], resultTypes, {}, [trueRegion, falseRegion]);
}

/**
 * Executes `whileOp`.
 * @param initArgs Parameter `initArgs`.
 * @param beforeRegion Parameter `beforeRegion`.
 * @param afterRegion Parameter `afterRegion`.
 * @param resultTypes Parameter `resultTypes`.
 * @returns Result of `whileOp`.
 */
export function whileOp(
  initArgs: Value[],
  beforeRegion: Region,
  afterRegion: Region,
  resultTypes: Type[],
): Operation {
  return new Operation('web.scf.while', initArgs, resultTypes, {}, [beforeRegion, afterRegion]);
}

/**
 * Executes `condition`.
 * @param cond Parameter `cond`.
 * @param args Parameter `args`.
 * @returns Result of `condition`.
 */
export function condition(cond: Value, args: Value[]): Operation {
  return new Operation('web.scf.condition', [cond, ...args], []);
}
