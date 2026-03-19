import { Operation, Type, Value, Region } from '../../ir/core.js';

// 92. web.scf dialect
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

export function yieldOp(results: Value[]): Operation {
  return new Operation('web.scf.yield', results, []);
}

export function ifOp(
  condition: Value,
  trueRegion: Region,
  falseRegion: Region,
  resultTypes: Type[],
): Operation {
  return new Operation('web.scf.if', [condition], resultTypes, {}, [trueRegion, falseRegion]);
}

export function whileOp(
  initArgs: Value[],
  beforeRegion: Region,
  afterRegion: Region,
  resultTypes: Type[],
): Operation {
  return new Operation('web.scf.while', initArgs, resultTypes, {}, [beforeRegion, afterRegion]);
}

export function condition(cond: Value, args: Value[]): Operation {
  return new Operation('web.scf.condition', [cond, ...args], []);
}
