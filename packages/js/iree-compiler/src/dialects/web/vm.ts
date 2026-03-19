import { Operation, Type, Value, Region, Block } from '../../ir/core.js';

// 71. web.vm.module
export function moduleOp(body: Region): Operation {
  return new Operation('web.vm.module', [], [], {}, [body]);
}

// 72. web.vm.func
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
export function call(callee: string, operands: Value[], resultTypes: Type[]): Operation {
  return new Operation('web.vm.call', operands, resultTypes, { callee });
}

// 74. web.vm.branch
export function branch(dest: Block, operands: Value[]): Operation {
  return new Operation('web.vm.branch', operands, [], { dest });
}

// 75. web.vm.cond_branch
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
export function cmp(pred: string, lhs: Value, rhs: Value, resultType: Type): Operation {
  return new Operation('web.vm.cmp', [lhs, rhs], [resultType], { predicate: pred });
}

// 77. web.vm arithmetic
export function addI32(lhs: Value, rhs: Value, resultType: Type): Operation {
  return new Operation('web.vm.add.i32', [lhs, rhs], [resultType]);
}

export function mulI32(lhs: Value, rhs: Value, resultType: Type): Operation {
  return new Operation('web.vm.mul.i32', [lhs, rhs], [resultType]);
}

// 78. web.vm.return
export function returnOp(operands: Value[]): Operation {
  return new Operation('web.vm.return', operands, []);
}

// 85. web.vm.import
export function importOp(name: string, moduleName: string, funcName: string): Operation {
  return new Operation('web.vm.import', [], [], { name, module: moduleName, function: funcName });
}
