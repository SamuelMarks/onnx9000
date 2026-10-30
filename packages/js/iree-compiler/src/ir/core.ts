/**
 * @fileoverview core.ts
 * Provides core functionality for the iree-compiler package.
 */
export type Type = {
  readonly id: string;
};

/** Represents the `Value` class. */
export class Value {
  /** Property `type` of `Value`. */
  readonly type: Type;
  /** Property `owner` of `Value`. */
  readonly owner: Operation | Block;

  /**
   * Initializes a new `Value` instance.
   * @param type Parameter `type`.
   * @param owner Parameter `owner`.
   */
  constructor(type: Type, owner: Operation | Block) {
    this.type = type;
    this.owner = owner;
  }
}

/** Represents the `BlockArgument` class. */
export class BlockArgument extends Value {
  /** Property `index` of `BlockArgument`. */
  readonly index: number;
  /** Property `parentBlock` of `BlockArgument`. */
  readonly parentBlock: Block;

  /**
   * Initializes a new `BlockArgument` instance.
   * @param type Parameter `type`.
   * @param index Parameter `index`.
   * @param parentBlock Parameter `parentBlock`.
   */
  constructor(type: Type, index: number, parentBlock: Block) {
    super(type, parentBlock);
    this.index = index;
    this.parentBlock = parentBlock;
  }
}

/** Represents the `Block` class. */
export class Block {
  /** Property `args` of `Block`. */
  readonly args: BlockArgument[] = [];
  /** Property `operations` of `Block`. */
  readonly operations: Operation[] = [];
  /** Property `parentRegion` of `Block`. */
  readonly parentRegion: Region | null;

  /**
   * Initializes a new `Block` instance.
   * @param parentRegion Parameter `parentRegion`.
   */
  constructor(parentRegion: Region | null = null) {
    this.parentRegion = parentRegion;
  }

  /**
   * Executes `addArgument` on `Block`.
   * @param type Parameter `type`.
   * @returns Result of `addArgument`.
   */
  addArgument(type: Type): BlockArgument {
    const arg = new BlockArgument(type, this.args.length, this);
    this.args.push(arg);
    return arg;
  }

  /**
   * Executes `pushOperation` on `Block`.
   * @param op Parameter `op`.
   */
  pushOperation(op: Operation): void {
    this.operations.push(op);
    op.parentBlock = this;
  }
}

/** Represents the `Region` class. */
export class Region {
  /** Property `blocks` of `Region`. */
  readonly blocks: Block[] = [];
  /** Property `parentOperation` of `Region`. */
  readonly parentOperation: Operation | null;

  /**
   * Initializes a new `Region` instance.
   * @param parentOperation Parameter `parentOperation`.
   */
  constructor(parentOperation: Operation | null = null) {
    this.parentOperation = parentOperation;
  }

  /**
   * Executes `pushBlock` on `Region`.
   * @param block Parameter `block`.
   */
  pushBlock(block: Block): void {
    this.blocks.push(block);
    // Normally block's parentRegion would be set here, but we pass it on construction.
  }
}

/** Represents the `Operation` class. */
export class Operation {
  /** Property `opcode` of `Operation`. */
  readonly opcode: string;
  /** Property `operands` of `Operation`. */
  readonly operands: Value[];
  /** Property `results` of `Operation`. */
  readonly results: Value[];
  /** Property `attributes` of `Operation`. */
  readonly attributes: Record<string, ReturnType<typeof JSON.parse>>;
  /** Property `regions` of `Operation`. */
  readonly regions: Region[];
  /** Property `parentBlock` of `Operation`. */
  parentBlock: Block | null = null;

  /**
   * Initializes a new `Operation` instance.
   * @param opcode Parameter `opcode`.
   * @param operands Parameter `operands`.
   * @param resultTypes Parameter `resultTypes`.
   * @param attributes Parameter `attributes`.
   * @param regions Parameter `regions`.
   */
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
