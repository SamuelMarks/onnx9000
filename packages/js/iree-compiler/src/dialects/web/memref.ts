import { Operation, Type, Value } from '../../ir/core.js';

export class MemRefType implements Type {
  readonly id = 'memref';
  readonly shape: number[];
  readonly elementType: string;

  constructor(shape: number[], elementType: string) {
    this.shape = [...shape];
    this.elementType = elementType;
  }
}

// 48. web.memref.alloc
export function alloc(type: MemRefType): Operation {
  return new Operation('web.memref.alloc', [], [type]);
}

// 49. web.memref.dealloc
export function dealloc(memref: Value): Operation {
  return new Operation('web.memref.dealloc', [memref], []);
}

// 50. web.memref.load
export function load(memref: Value, indices: Value[], resultType: Type): Operation {
  return new Operation('web.memref.load', [memref, ...indices], [resultType]);
}

// 50. web.memref.store
export function store(value: Value, memref: Value, indices: Value[]): Operation {
  return new Operation('web.memref.store', [value, memref, ...indices], []);
}
