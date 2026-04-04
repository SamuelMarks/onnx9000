import { Tensor } from '../ir/tensor.js';
import { ConvND, Gemm, RMSNorm, Silu } from '../primitives.js';

function getParam(name: string, shape: number[], dtype: any = 'float32'): Tensor {
  return new Tensor(name, shape, dtype, false, false, new Float32Array());
}

function recordOp(opType: string, inputs: Tensor[], attr?: any): Tensor {
  const dtype = inputs[0]?.dtype ?? 'float32';
  return new Tensor(`${opType}_out`, [], dtype, false, false, new Float32Array());
}

export class StateSpace {
  public dModel: number;
  public dState: number;
  public dConv: number;
  public expand: number;

  constructor(dModel: number, dState: number, dConv: number, expand: number = 2) {
    this.dModel = dModel;
    this.dState = dState;
    this.dConv = dConv;
    this.expand = expand;
  }

  call(x: Tensor, dt: Tensor, a: Tensor, b: Tensor, c: Tensor, d: Tensor): Tensor {
    return recordOp('StateSpace', [x, dt, a, b, c, d], {
      d_model: this.dModel,
      d_state: this.dState,
      d_conv: this.dConv,
      expand: this.expand,
    });
  }
}

export class MambaBlock {
  public prefix: string;
  public dModel: number;
  public dState: number;
  public dConv: number;
  public expand: number;

  public norm: RMSNorm;
  public inProj: Gemm;
  public conv1d: ConvND;
  public act: Silu;
  public xProj: Gemm;
  public dtProj: Gemm;
  public ssm: StateSpace;
  public outProj: Gemm;

  constructor(dModel: number, dState: number, dConv: number, expand: number, prefix: string = '') {
    this.prefix = prefix;
    this.dModel = dModel;
    this.dState = dState;
    this.dConv = dConv;
    this.expand = expand;

    this.norm = new RMSNorm([dModel]);
    this.inProj = new Gemm(1.0, 1.0, 0, 1);

    const dInner = dModel * expand;
    this.conv1d = new ConvND(1, dInner, dInner, dConv, 1, dConv - 1, 1, dInner, false);
    this.act = new Silu();

    this.xProj = new Gemm(1.0, 1.0, 0, 1);
    this.dtProj = new Gemm(1.0, 1.0, 0, 1);

    this.ssm = new StateSpace(dModel, dState, dConv, expand);

    this.outProj = new Gemm(1.0, 1.0, 0, 1);
  }

  call(x: Tensor): Tensor {
    const identity = x;
    const xNorm = this.norm.call(x, getParam(`${this.prefix}.norm.weight`, [this.dModel]));

    const dInner = this.dModel * this.expand;
    const xz = this.inProj.call(
      xNorm,
      getParam(`${this.prefix}.in_proj.weight`, [dInner * 2, this.dModel]),
    );

    const xInner = recordOp('Slice', [
      xz,
      recordOp('Constant', [], { value: [0], dtype: 7 }),
      recordOp('Constant', [], { value: [dInner], dtype: 7 }),
      recordOp('Constant', [], { value: [2], dtype: 7 }),
    ]);
    const z = recordOp('Slice', [
      xz,
      recordOp('Constant', [], { value: [dInner], dtype: 7 }),
      recordOp('Constant', [], { value: [dInner * 2], dtype: 7 }),
      recordOp('Constant', [], { value: [2], dtype: 7 }),
    ]);

    let xInnerT = recordOp('Transpose', [xInner], { perm: [0, 2, 1] });
    let xInnerConv = this.conv1d.call(
      xInnerT,
      getParam(`${this.prefix}.conv1d.weight`, [dInner, 1, this.dConv]),
      getParam(`${this.prefix}.conv1d.bias`, [dInner]),
    );
    xInnerConv = recordOp('Transpose', [xInnerConv], { perm: [0, 2, 1] });

    const seqLen = recordOp(
      'Gather',
      [recordOp('Shape', [x]), recordOp('Constant', [], { value: [1], dtype: 7 })],
      { axis: 0 },
    );
    xInnerConv = recordOp('Slice', [
      xInnerConv,
      recordOp('Constant', [], { value: [0], dtype: 7 }),
      seqLen,
      recordOp('Constant', [], { value: [1], dtype: 7 }),
    ]);

    const xAct = this.act.call(xInnerConv);

    const xDtBC = this.xProj.call(
      xAct,
      getParam(`${this.prefix}.x_proj.weight`, [this.dState * 2 + 1, dInner]),
    );

    let dt = recordOp('Slice', [
      xDtBC,
      recordOp('Constant', [], { value: [0], dtype: 7 }),
      recordOp('Constant', [], { value: [1], dtype: 7 }),
      recordOp('Constant', [], { value: [2], dtype: 7 }),
    ]);
    const b = recordOp('Slice', [
      xDtBC,
      recordOp('Constant', [], { value: [1], dtype: 7 }),
      recordOp('Constant', [], { value: [this.dState + 1], dtype: 7 }),
      recordOp('Constant', [], { value: [2], dtype: 7 }),
    ]);
    const c = recordOp('Slice', [
      xDtBC,
      recordOp('Constant', [], { value: [this.dState + 1], dtype: 7 }),
      recordOp('Constant', [], { value: [this.dState * 2 + 1], dtype: 7 }),
      recordOp('Constant', [], { value: [2], dtype: 7 }),
    ]);

    dt = this.dtProj.call(
      dt,
      getParam(`${this.prefix}.dt_proj.weight`, [dInner, 1]),
      getParam(`${this.prefix}.dt_proj.bias`, [dInner]),
    );

    const a = getParam(`${this.prefix}.A_log`, [dInner, this.dState]);
    const d = getParam(`${this.prefix}.D`, [dInner]);

    let y = this.ssm.call(xAct, dt, a, b, c, d);

    const zAct = this.act.call(z);
    y = recordOp('Mul', [y, zAct]);

    const out = this.outProj.call(
      y,
      getParam(`${this.prefix}.out_proj.weight`, [this.dModel, dInner]),
    );
    return recordOp('Add', [identity, out]);
  }
}

export class Mamba {
  public vocabSize: number;
  public dModel: number;
  public blocks: MambaBlock[];
  public norm: RMSNorm;
  public lmHead: Gemm;

  constructor(
    vocabSize: number = 50277,
    dModel: number = 768,
    nLayer: number = 24,
    dState: number = 16,
    dConv: number = 4,
    expand: number = 2,
  ) {
    this.vocabSize = vocabSize;
    this.dModel = dModel;

    this.blocks = [];
    for (let i = 0; i < nLayer; i++) {
      this.blocks.push(new MambaBlock(dModel, dState, dConv, expand, `blocks.${i}`));
    }

    this.norm = new RMSNorm([dModel]);
    this.lmHead = new Gemm(1.0, 1.0, 0, 1);
  }

  call(inputIds: Tensor): Tensor {
    let x = recordOp(
      'Gather',
      [getParam('embedding.weight', [this.vocabSize, this.dModel]), inputIds],
      { axis: 0 },
    );

    for (const block of this.blocks) {
      x = block.call(x);
    }

    x = this.norm.call(x, getParam('norm.weight', [this.dModel]));
    x = this.lmHead.call(x, getParam('lm_head.weight', [this.vocabSize, this.dModel]));
    return x;
  }
}

export function mamba130m(): Mamba {
  return new Mamba(50277, 768, 24, 16, 4, 2);
}
