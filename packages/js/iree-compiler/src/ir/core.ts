export type Type = {
  readonly id: string;
};

export class Value {
  readonly type: Type;
  readonly owner: Operation | Block;

  constructor(type: Type, owner: Operation | Block) {
    this.type = type;
    this.owner = owner;
  }
}

export class BlockArgument extends Value {
  readonly index: number;
  readonly parentBlock: Block;

  constructor(type: Type, index: number, parentBlock: Block) {
    super(type, parentBlock);
    this.index = index;
    this.parentBlock = parentBlock;
  }
}

export class Block {
  readonly args: BlockArgument[] = [];
  readonly operations: Operation[] = [];
  readonly parentRegion: Region | null;

  constructor(parentRegion: Region | null = null) {
    this.parentRegion = parentRegion;
  }

  addArgument(type: Type): BlockArgument {
    const arg = new BlockArgument(type, this.args.length, this);
    this.args.push(arg);
    return arg;
  }

  pushOperation(op: Operation): void {
    this.operations.push(op);
    op.parentBlock = this;
  }
}

export class Region {
  readonly blocks: Block[] = [];
  readonly parentOperation: Operation | null;

  constructor(parentOperation: Operation | null = null) {
    this.parentOperation = parentOperation;
  }

  pushBlock(block: Block): void {
    this.blocks.push(block);
    // Normally block's parentRegion would be set here, but we pass it on construction.
  }
}

export class Operation {
  readonly opcode: string;
  readonly operands: Value[];
  readonly results: Value[];
  readonly attributes: Record<string, ReturnType<typeof JSON.parse>>;
  readonly regions: Region[];
  parentBlock: Block | null = null;

  constructor(
    opcode: string,
    operands: Value[] = [],
    resultTypes: Type[] = [],
    attributes: Record<string, ReturnType<typeof JSON.parse>> = {},
    regions: Region[] = [],
  ) {
    this.opcode = opcode;
    this.operands = [...operands];
    this.attributes = { ...attributes };
    this.regions = [...regions];
    for (const region of this.regions) {
      // Re-assigning the operation, though typescript makes it readonly in Region class, we can workaround if needed
      (region as ReturnType<typeof JSON.parse>).parentOperation = this;
    }

    this.results = resultTypes.map((t) => new Value(t, this));
  }
}
