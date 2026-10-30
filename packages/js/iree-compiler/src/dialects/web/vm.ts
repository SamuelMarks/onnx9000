/**
 * @fileoverview vm.ts
 * Provides vm functionality for the iree-compiler package.
 */
import { type Block, Operation, type Region, type Type, type Value } from '../../ir/core.js';

// 71. web.vm.module
/**
 * Executes `moduleOp`.
 * @param body Parameter `body`.
 * @returns Result of `moduleOp`.
 */
export function moduleOp(body: Region): Operation {
  return new Operation('web.vm.module', [], [], {}, [body]);
}

// 72. web.vm.func
/**
 * Executes `func`.
 * @param name Parameter `name`.
 * @param argTypes Parameter `argTypes`.
 * @param resultTypes Parameter `resultTypes`.
 * @param body Parameter `body`.
 * @returns Result of `func`.
 */
export function func(name: string, argTypes: Type[], resultTypes: Type[], body: Region): Operation {
  return new Operation(
    'web.vm.func',
    [],
    [],
    {
      name,
      arg_types: argTypes,
      result_types: resultTypes,
    },
    [body],
  );
}

// 73. web.vm.call
/**
 * Executes `call`.
 * @param callee Parameter `callee`.
 * @param operands Parameter `operands`.
 * @param resultTypes Parameter `resultTypes`.
 * @returns Result of `call`.
 */
export function call(callee: string, operands: Value[], resultTypes: Type[]): Operation {
  return new Operation('web.vm.call', operands, resultTypes, { callee });
}

// 74. web.vm.branch
/**
 * Executes `branch`.
 * @param dest Parameter `dest`.
 * @param operands Parameter `operands`.
 * @returns Result of `branch`.
 */
export function branch(dest: Block, operands: Value[]): Operation {
  return new Operation('web.vm.branch', operands, [], { dest });
}

// 75. web.vm.cond_branch
/**
 * Executes `condBranch`.
 * @param cond Parameter `cond`.
 * @param trueDest Parameter `trueDest`.
 * @param trueOperands Parameter `trueOperands`.
 * @param falseDest Parameter `falseDest`.
 * @param falseOperands Parameter `falseOperands`.
 * @returns Result of `condBranch`.
 */
export function condBranch(
  cond: Value,
  trueDest: Block,
  trueOperands: Value[],
  falseDest: Block,
  falseOperands: Value[],
): Operation {
  return new Operation('web.vm.cond_branch', [cond, ...trueOperands, ...falseOperands], [], {
    true_dest: trueDest,
    false_dest: falseDest,
    true_operands_count: trueOperands.length,
  });
}

// 76. web.vm.cmp
/**
 * Executes `cmp`.
 * @param pred Parameter `pred`.
 * @param lhs Parameter `lhs`.
 * @param rhs Parameter `rhs`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `cmp`.
 */
export function cmp(pred: string, lhs: Value, rhs: Value, resultType: Type): Operation {
  return new Operation('web.vm.cmp', [lhs, rhs], [resultType], {
    predicate: pred,
  });
}

// 77. web.vm arithmetic
/**
 * Executes `addI32`.
 * @param lhs Parameter `lhs`.
 * @param rhs Parameter `rhs`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `addI32`.
 */
export function addI32(lhs: Value, rhs: Value, resultType: Type): Operation {
  return new Operation('web.vm.add.i32', [lhs, rhs], [resultType]);
}

/**
 * Executes `mulI32`.
 * @param lhs Parameter `lhs`.
 * @param rhs Parameter `rhs`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `mulI32`.
 */
export function mulI32(lhs: Value, rhs: Value, resultType: Type): Operation {
  return new Operation('web.vm.mul.i32', [lhs, rhs], [resultType]);
}

// 78. web.vm.return
/**
 * Executes `returnOp`.
 * @param operands Parameter `operands`.
 * @returns Result of `returnOp`.
 */
export function returnOp(operands: Value[]): Operation {
  return new Operation('web.vm.return', operands, []);
}

// 85. web.vm.import
/**
 * Executes `importOp`.
 * @param name Parameter `name`.
 * @param moduleName Parameter `moduleName`.
 * @param funcName Parameter `funcName`.
 * @returns Result of `importOp`.
 */
export function importOp(name: string, moduleName: string, funcName: string): Operation {
  return new Operation('web.vm.import', [], [], {
    name,
    module: moduleName,
    function: funcName,
  });
}
