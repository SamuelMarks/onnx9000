import { Tensor as CoreTensor, DType, Shape } from '@onnx9000/core';

export class BaseTensor extends CoreTensor {
  /** The node type in the ONNX graph if lazy */
  opType?: string;
  /** The inputs to this operation */
  inputs: BaseTensor[];

  constructor(
    name: string,
    shape: Shape,
    dtype: DType,
    opType?: string,
    inputs: BaseTensor[] = [],
  ) {
    super(name, shape, dtype, false, true, null);
    this.opType = opType;
    this.inputs = inputs;
  }
}

export class EagerTensor extends BaseTensor {
  constructor(data: any, dtype: DType = 'float32') {
    super('eager', [data?.length || 0], dtype);
    this.data = data;
  }

  get ndim(): number {
    return this.shape.length;
  }

  /** Returns raw data */
  numpy(): any {
    return this.data;
  }
  data_val(): any {
    return this.data;
  }

  /** Eager Evaluation method for AST wrapper */
  evaluate(): EagerTensor {
    // Mock evaluation logic bridging Eager and Lazy contexts
    return this;
  }

  dispose(): void {
    this.data = null;
  }

  cpu(): EagerTensor {
    return this;
  }
  gpu(): EagerTensor {
    return this;
  }
  quantize_dynamic(): EagerTensor {
    return this;
  }

  get T(): EagerTensor {
    return transpose(this);
  }

  /** add operation */
  add(b: EagerTensor | number): EagerTensor {
    return add(this, b);
  }

  /** subtract operation */
  subtract(b: EagerTensor | number): EagerTensor {
    return subtract(this, b);
  }

  /** multiply operation */
  multiply(b: EagerTensor | number): EagerTensor {
    return multiply(this, b);
  }

  /** divide operation */
  divide(b: EagerTensor | number): EagerTensor {
    return divide(this, b);
  }

  /** power operation */
  power(b: EagerTensor | number): EagerTensor {
    return power(this, b);
  }

  /** mod operation */
  mod(b: EagerTensor | number): EagerTensor {
    return mod(this, b);
  }

  /** absolute operation */
  absolute(): EagerTensor {
    return absolute(this);
  }

  /** negative operation */
  negative(): EagerTensor {
    return negative(this);
  }

  /** sign operation */
  sign(): EagerTensor {
    return sign(this);
  }

  /** exp operation */
  exp(): EagerTensor {
    return exp(this);
  }

  /** log operation */
  log(): EagerTensor {
    return log(this);
  }

  /** sqrt operation */
  sqrt(): EagerTensor {
    return sqrt(this);
  }

  /** square operation */
  square(b: EagerTensor | number): EagerTensor {
    return square(this, b);
  }

  /** sin operation */
  sin(): EagerTensor {
    return sin(this);
  }

  /** cos operation */
  cos(): EagerTensor {
    return cos(this);
  }

  /** tan operation */
  tan(): EagerTensor {
    return tan(this);
  }

  /** arcsin operation */
  arcsin(): EagerTensor {
    return arcsin(this);
  }

  /** arccos operation */
  arccos(): EagerTensor {
    return arccos(this);
  }

  /** arctan operation */
  arctan(): EagerTensor {
    return arctan(this);
  }

  /** sinh operation */
  sinh(): EagerTensor {
    return sinh(this);
  }

  /** cosh operation */
  cosh(): EagerTensor {
    return cosh(this);
  }

  /** tanh operation */
  tanh(): EagerTensor {
    return tanh(this);
  }

  /** arcsinh operation */
  arcsinh(): EagerTensor {
    return arcsinh(this);
  }

  /** arccosh operation */
  arccosh(): EagerTensor {
    return arccosh(this);
  }

  /** arctanh operation */
  arctanh(): EagerTensor {
    return arctanh(this);
  }

  /** matmul operation */
  matmul(b: EagerTensor | number): EagerTensor {
    return matmul(this, b);
  }

  /** equal operation */
  equal(b: EagerTensor | number): EagerTensor {
    return equal(this, b);
  }

  /** less operation */
  less(b: EagerTensor | number): EagerTensor {
    return less(this, b);
  }

  /** greater operation */
  greater(b: EagerTensor | number): EagerTensor {
    return greater(this, b);
  }

  /** less_equal operation */
  less_equal(b: EagerTensor | number): EagerTensor {
    return less_equal(this, b);
  }

  /** greater_equal operation */
  greater_equal(b: EagerTensor | number): EagerTensor {
    return greater_equal(this, b);
  }

  /** logical_and operation */
  logical_and(b: EagerTensor | number): EagerTensor {
    return logical_and(this, b);
  }

  /** logical_or operation */
  logical_or(b: EagerTensor | number): EagerTensor {
    return logical_or(this, b);
  }

  /** logical_not operation */
  logical_not(): EagerTensor {
    return logical_not(this);
  }

  /** logical_xor operation */
  logical_xor(b: EagerTensor | number): EagerTensor {
    return logical_xor(this, b);
  }

  /** isnan operation */
  isnan(): EagerTensor {
    return isnan(this);
  }

  /** isinf operation */
  isinf(): EagerTensor {
    return isinf(this);
  }
}

export class LazyTensor extends BaseTensor {
  constructor(opType: string, inputs: BaseTensor[], dtype: DType = 'float32') {
    super('lazy_' + opType, [], dtype, opType, inputs);
  }
}

export let IS_LAZY = false;
export function lazy_mode(enable: boolean): void {
  IS_LAZY = enable;
}
export function Input(name: string, shape: Shape, dtype: DType): LazyTensor {
  return new LazyTensor('Input', [], dtype);
}

export function array(data: any, dtype: DType = 'float32'): EagerTensor {
  return new EagerTensor(data, dtype);
}

/** Functional add */
export function add(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Add', args);
  return new EagerTensor(null);
}

/** Functional subtract */
export function subtract(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Sub', args);
  return new EagerTensor(null);
}

/** Functional multiply */
export function multiply(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Mul', args);
  return new EagerTensor(null);
}

/** Functional divide */
export function divide(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Div', args);
  return new EagerTensor(null);
}

/** Functional power */
export function power(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Pow', args);
  return new EagerTensor(null);
}

/** Functional mod */
export function mod(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Mod', args);
  return new EagerTensor(null);
}

/** Functional absolute */
export function absolute(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Abs', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional negative */
export function negative(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Neg', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional sign */
export function sign(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Sign', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional exp */
export function exp(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Exp', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional log */
export function log(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Log', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional sqrt */
export function sqrt(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Sqrt', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional square */
export function square(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Mul', args);
  return new EagerTensor(null);
}

/** Functional sin */
export function sin(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Sin', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional cos */
export function cos(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Cos', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional tan */
export function tan(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Tan', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional arcsin */
export function arcsin(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Asin', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional arccos */
export function arccos(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Acos', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional arctan */
export function arctan(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Atan', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional sinh */
export function sinh(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Sinh', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional cosh */
export function cosh(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Cosh', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional tanh */
export function tanh(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Tanh', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional arcsinh */
export function arcsinh(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Asinh', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional arccosh */
export function arccosh(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Acosh', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional arctanh */
export function arctanh(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Atanh', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional matmul */
export function matmul(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('MatMul', args);
  return new EagerTensor(null);
}

/** Functional equal */
export function equal(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Equal', args);
  return new EagerTensor(null);
}

/** Functional less */
export function less(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Less', args);
  return new EagerTensor(null);
}

/** Functional greater */
export function greater(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Greater', args);
  return new EagerTensor(null);
}

/** Functional less_equal */
export function less_equal(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('LessOrEqual', args);
  return new EagerTensor(null);
}

/** Functional greater_equal */
export function greater_equal(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('GreaterOrEqual', args);
  return new EagerTensor(null);
}

/** Functional logical_and */
export function logical_and(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('And', args);
  return new EagerTensor(null);
}

/** Functional logical_or */
export function logical_or(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Or', args);
  return new EagerTensor(null);
}

/** Functional logical_not */
export function logical_not(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Not', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional logical_xor */
export function logical_xor(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Xor', args);
  return new EagerTensor(null);
}

/** Functional isnan */
export function isnan(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('IsNaN', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional isinf */
export function isinf(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('IsInf', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional where */
export function where(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Where', args);
  return new EagerTensor(null);
}

/** Functional sum */
export function sum(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('ReduceSum', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional prod */
export function prod(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('ReduceProd', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional mean */
export function mean(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('ReduceMean', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional min */
export function min(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('ReduceMin', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional max */
export function max(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('ReduceMax', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional argmin */
export function argmin(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('ArgMin', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional argmax */
export function argmax(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('ArgMax', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional reshape */
export function reshape(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Reshape', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional squeeze */
export function squeeze(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Squeeze', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional expand_dims */
export function expand_dims(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Unsqueeze', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional concatenate */
export function concatenate(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Concat', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional split */
export function split(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Split', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional tile */
export function tile(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Tile', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional pad */
export function pad(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Pad', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional transpose */
export function transpose(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Transpose', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional take */
export function take(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Gather', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional gather */
export function gather(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Gather', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional sort */
export function sort(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('Sort', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional argsort */
export function argsort(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('ArgSort', [a, ...args]);
  return new EagerTensor(null);
}

/** Functional nonzero */
export function nonzero(a: any, ...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('NonZero', [a, ...args]);
  return new EagerTensor(null);
}

/** Function zeros */
export function zeros(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('zeros', args);
  return new EagerTensor(null);
}

/** Function ones */
export function ones(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('ones', args);
  return new EagerTensor(null);
}

/** Function empty */
export function empty(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('empty', args);
  return new EagerTensor(null);
}

/** Function full */
export function full(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('full', args);
  return new EagerTensor(null);
}

/** Function eye */
export function eye(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('eye', args);
  return new EagerTensor(null);
}

/** Function identity */
export function identity(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('identity', args);
  return new EagerTensor(null);
}

/** Function arange */
export function arange(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('arange', args);
  return new EagerTensor(null);
}

/** Function linspace */
export function linspace(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('linspace', args);
  return new EagerTensor(null);
}

/** Function log10 */
export function log10(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('log10', args);
  return new EagerTensor(null);
}

/** Function log2 */
export function log2(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('log2', args);
  return new EagerTensor(null);
}

/** Function cbrt */
export function cbrt(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('cbrt', args);
  return new EagerTensor(null);
}

/** Function reciprocal */
export function reciprocal(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('reciprocal', args);
  return new EagerTensor(null);
}

/** Function deg2rad */
export function deg2rad(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('deg2rad', args);
  return new EagerTensor(null);
}

/** Function rad2deg */
export function rad2deg(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('rad2deg', args);
  return new EagerTensor(null);
}

/** Function dot */
export function dot(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('dot', args);
  return new EagerTensor(null);
}

/** Function vdot */
export function vdot(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('vdot', args);
  return new EagerTensor(null);
}

/** Function inner */
export function inner(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('inner', args);
  return new EagerTensor(null);
}

/** Function outer */
export function outer(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('outer', args);
  return new EagerTensor(null);
}

/** Function tensordot */
export function tensordot(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('tensordot', args);
  return new EagerTensor(null);
}

/** Function einsum */
export function einsum(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('einsum', args);
  return new EagerTensor(null);
}

/** Function swapaxes */
export function swapaxes(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('swapaxes', args);
  return new EagerTensor(null);
}

/** Function trace */
export function trace(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('trace', args);
  return new EagerTensor(null);
}

/** Function ptp */
export function ptp(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('ptp', args);
  return new EagerTensor(null);
}

/** Function all */
export function all(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('all', args);
  return new EagerTensor(null);
}

/** Function any */
export function any(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('any', args);
  return new EagerTensor(null);
}

/** Function cumsum */
export function cumsum(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('cumsum', args);
  return new EagerTensor(null);
}

/** Function cumprod */
export function cumprod(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('cumprod', args);
  return new EagerTensor(null);
}

/** Function ravel */
export function ravel(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('ravel', args);
  return new EagerTensor(null);
}

/** Function broadcast_to */
export function broadcast_to(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('broadcast_to', args);
  return new EagerTensor(null);
}

/** Function stack */
export function stack(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('stack', args);
  return new EagerTensor(null);
}

/** Function vstack */
export function vstack(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('vstack', args);
  return new EagerTensor(null);
}

/** Function hstack */
export function hstack(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('hstack', args);
  return new EagerTensor(null);
}

/** Function dstack */
export function dstack(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('dstack', args);
  return new EagerTensor(null);
}

/** Function array_split */
export function array_split(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('array_split', args);
  return new EagerTensor(null);
}

/** Function repeat */
export function repeat(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('repeat', args);
  return new EagerTensor(null);
}

/** Function not_equal */
export function not_equal(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('not_equal', args);
  return new EagerTensor(null);
}

/** Function allclose */
export function allclose(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('allclose', args);
  return new EagerTensor(null);
}

/** Function isclose */
export function isclose(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('isclose', args);
  return new EagerTensor(null);
}

/** Function extract */
export function extract(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('extract', args);
  return new EagerTensor(null);
}

/** Function take_along_axis */
export function take_along_axis(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('take_along_axis', args);
  return new EagerTensor(null);
}

/** Function put */
export function put(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('put', args);
  return new EagerTensor(null);
}

/** Function put_along_axis */
export function put_along_axis(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('put_along_axis', args);
  return new EagerTensor(null);
}

/** Function nan_to_num */
export function nan_to_num(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('nan_to_num', args);
  return new EagerTensor(null);
}

/** Function clip */
export function clip(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('clip', args);
  return new EagerTensor(null);
}

/** Function around */
export function around(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('around', args);
  return new EagerTensor(null);
}

/** Function fix */
export function fix(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('fix', args);
  return new EagerTensor(null);
}

/** Function i0 */
export function i0(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('i0', args);
  return new EagerTensor(null);
}

/** Function sinc */
export function sinc(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('sinc', args);
  return new EagerTensor(null);
}

/** Function save */
export function save(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('save', args);
  return new EagerTensor(null);
}

/** Function load */
export function load(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('load', args);
  return new EagerTensor(null);
}

/** Function vectorize */
export function vectorize(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('vectorize', args);
  return new EagerTensor(null);
}

/** Function meshgrid */
export function meshgrid(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('meshgrid', args);
  return new EagerTensor(null);
}

/** Function mgrid */
export function mgrid(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('mgrid', args);
  return new EagerTensor(null);
}

/** Function einsum_path */
export function einsum_path(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('einsum_path', args);
  return new EagerTensor(null);
}

/** Function polyfit */
export function polyfit(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('polyfit', args);
  return new EagerTensor(null);
}

/** Function histogram */
export function histogram(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('histogram', args);
  return new EagerTensor(null);
}

/** Function digitize */
export function digitize(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('digitize', args);
  return new EagerTensor(null);
}

/** Function export_model */
export function export_model(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('export_model', args);
  return new EagerTensor(null);
}

/** Function compile */
export function compile(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('compile', args);
  return new EagerTensor(null);
}

/** Function set_device */
export function set_device(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('set_device', args);
  return new EagerTensor(null);
}

/** Function set_log_level */
export function set_log_level(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('set_log_level', args);
  return new EagerTensor(null);
}

/** Function set_opset */
export function set_opset(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('set_opset', args);
  return new EagerTensor(null);
}

/** Function set_num_threads */
export function set_num_threads(...args: any[]): any {
  if (IS_LAZY) return new LazyTensor('set_num_threads', args);
  return new EagerTensor(null);
}

export const nn = {
  relu: (x: any) => (IS_LAZY ? new LazyTensor('Relu', [x]) : new EagerTensor(null)),
  sigmoid: (x: any) => (IS_LAZY ? new LazyTensor('Sigmoid', [x]) : new EagerTensor(null)),
  softmax: (x: any, axis: any = -1) =>
    IS_LAZY ? new LazyTensor('Softmax', [x, axis]) : new EagerTensor(null),
  log_softmax: (x: any, axis: any = -1) =>
    IS_LAZY ? new LazyTensor('LogSoftmax', [x, axis]) : new EagerTensor(null),
  gelu: (x: any) => (IS_LAZY ? new LazyTensor('Gelu', [x]) : new EagerTensor(null)),
  conv2d: (...args: any[]) => (IS_LAZY ? new LazyTensor('Conv', args) : new EagerTensor(null)),
  max_pool2d: (...args: any[]) =>
    IS_LAZY ? new LazyTensor('MaxPool', args) : new EagerTensor(null),
  avg_pool2d: (...args: any[]) =>
    IS_LAZY ? new LazyTensor('AveragePool', args) : new EagerTensor(null),
  batch_norm: (...args: any[]) =>
    IS_LAZY ? new LazyTensor('BatchNormalization', args) : new EagerTensor(null),
  layer_norm: (...args: any[]) =>
    IS_LAZY ? new LazyTensor('LayerNormalization', args) : new EagerTensor(null),
  dropout: (...args: any[]) => (IS_LAZY ? new LazyTensor('Dropout', args) : new EagerTensor(null)),
  linear: (...args: any[]) => (IS_LAZY ? new LazyTensor('MatMul', args) : new EagerTensor(null)),
  cross_entropy_loss: (...args: any[]) =>
    IS_LAZY ? new LazyTensor('SoftmaxCrossEntropyLoss', args) : new EagerTensor(null),
};

export const linalg = {
  norm: (...args: any[]) =>
    IS_LAZY ? new LazyTensor('LpNormalization', args) : new EagerTensor(null),
  det: (...args: any[]) => (IS_LAZY ? new LazyTensor('Det', args) : new EagerTensor(null)),
  inv: (...args: any[]) => (IS_LAZY ? new LazyTensor('Inv', args) : new EagerTensor(null)),
  solve: (...args: any[]) => (IS_LAZY ? new LazyTensor('Solve', args) : new EagerTensor(null)),
  svd: (...args: any[]) => (IS_LAZY ? new LazyTensor('Svd', args) : new EagerTensor(null)),
};

export const char = {
  add: (...args: any[]) => (IS_LAZY ? new LazyTensor('StringConcat', args) : new EagerTensor(null)),
  equal: (...args: any[]) =>
    IS_LAZY ? new LazyTensor('StringEqual', args) : new EagerTensor(null),
  replace: (...args: any[]) =>
    IS_LAZY ? new LazyTensor('StringReplace', args) : new EagerTensor(null),
};

export const random = {
  rand: (...args: any[]) =>
    IS_LAZY ? new LazyTensor('RandomUniform', args) : new EagerTensor(null),
  randn: (...args: any[]) =>
    IS_LAZY ? new LazyTensor('RandomNormal', args) : new EagerTensor(null),
  randint: (...args: any[]) =>
    IS_LAZY ? new LazyTensor('RandomUniformInt', args) : new EagerTensor(null),
  uniform: (...args: any[]) =>
    IS_LAZY ? new LazyTensor('RandomUniform', args) : new EagerTensor(null),
  normal: (...args: any[]) =>
    IS_LAZY ? new LazyTensor('RandomNormal', args) : new EagerTensor(null),
  seed: (s: any) => {},
};

export class BroadcastError extends Error {}
export class TypeMismatchError extends Error {}
