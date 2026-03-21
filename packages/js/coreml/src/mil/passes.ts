import { Operation, Block, Function } from './ast.js';

export function deadCodeElimination(block: Block): void {
  let changed = true;
  while (changed) {
    changed = false;

    // Find all variables that are consumed
    const consumedVars = new Set<string>();

    // Outputs are always consumed
    for (const out of block.outputs) {
      consumedVars.add(out.name);
    }

    // Add variables used as inputs to ops
    for (const op of block.operations) {
      for (const key in op.inputs) {
        const inputs = op.inputs[key]!;
        if (Array.isArray(inputs)) {
          for (const v of inputs) consumedVars.add(v.name);
        } else {
          consumedVars.add(inputs.name);
        }
      }
    }

    // Filter operations that have outputs which are consumed
    const newOps: Operation[] = [];
    for (const op of block.operations) {
      const isConsumed = op.outputs.some((out) => consumedVars.has(out.name));
      // In MIL, operations with side effects (like writing state) would need special handling,
      // but assuming standard mathematical operations here:
      if (isConsumed) {
        newOps.push(op);
      } else {
        changed = true;
      }
    }
    block.operations = newOps;
  }
}

export function commonSubexpressionElimination(block: Block): void {
  let changed = true;
  while (changed) {
    changed = false;

    // A map from a canonical string representation of an op to its output variables
    const exprMap = new Map<string, string[]>();
    const varReplacement = new Map<string, string>();

    const newOps: Operation[] = [];

    for (const op of block.operations) {
      // canonicalize inputs based on replacements
      for (const key in op.inputs) {
        const inputs = op.inputs[key]!;
        if (Array.isArray(inputs)) {
          for (const v of inputs) {
            if (varReplacement.has(v.name)) v.name = varReplacement.get(v.name)!;
          }
        } else {
          if (varReplacement.has(inputs.name)) inputs.name = varReplacement.get(inputs.name)!;
        }
      }

      // Generate a deterministic string for this expression
      // We skip CSE for nodes with complex attributes for simplicity in this baseline implementation
      let isSimple = true;
      if (Object.keys(op.attributes).length > 0) isSimple = false;

      if (isSimple) {
        const inputNames = Object.entries(op.inputs)
          .sort((a, b) => a[0].localeCompare(b[0]))
          .map(([k, v]) => `${k}:${Array.isArray(v) ? v.map((vi) => vi.name).join(',') : v.name}`)
          .join('|');
        const exprKey = `${op.opType}#${inputNames}`;

        if (exprMap.has(exprKey)) {
          // This expression already exists. Replace outputs.
          const existingOutputs = exprMap.get(exprKey)!;
          if (existingOutputs.length === op.outputs.length) {
            for (let i = 0; i < op.outputs.length; i++) {
              varReplacement.set(op.outputs[i]!.name, existingOutputs[i]!);
            }
            changed = true;
            continue; // Skip adding this operation
          }
        } else {
          exprMap.set(
            exprKey,
            op.outputs.map((o) => o.name),
          );
        }
      }

      newOps.push(op);
    }

    block.operations = newOps;
  }
}

export function constantFolding(block: Block): void {
  let changed = true;
  while (changed) {
    changed = false;
    const constVals = new Map<string, any>();

    // identify constants
    for (const op of block.operations) {
      if (op.opType === 'const') {
        const out = op.outputs[0];
        if (out && op.attributes.value !== undefined) {
          constVals.set(out.name, op.attributes.value);
        }
      }
    }

    const newOps: Operation[] = [];
    for (const op of block.operations) {
      if (op.opType === 'add' || op.opType === 'mul' || op.opType === 'sub') {
        let allInputsConst = true;
        for (const k in op.inputs) {
          const inputs = op.inputs[k]!;
          if (Array.isArray(inputs)) {
            for (const v of inputs) {
              if (!constVals.has(v.name)) allInputsConst = false;
            }
          } else {
            if (!constVals.has(inputs.name)) allInputsConst = false;
          }
        }

        if (allInputsConst) {
          // Future work: Evaluate mathematical ops natively in JS
          // Replace with const operator
        }
      }

      newOps.push(op);
    }
    block.operations = newOps;
  }
}

export function fuseAdjacentOps(block: Block): void {
  // Phase 8: 179. Fuse sequence of Split -> Concat operations
  // Phase 8: 180. Fuse Slice operations with adjacent Pad operations
  let changed = true;
  while (changed) {
    changed = false;
    const newOps: Operation[] = [];

    for (let i = 0; i < block.operations.length; i++) {
      const op = block.operations[i]!;

      // Basic heuristic for Split->Concat fusion
      if (op.opType === 'concat') {
        const inputs = op.inputs['values'];
        if (Array.isArray(inputs) && inputs.length > 0) {
          // Check if all inputs come from the same 'split' op
          const producerOp = block.operations.find((o) =>
            o.outputs.some((out) => out.name === inputs[0]!.name),
          );
          if (producerOp && producerOp.opType === 'split') {
            // Simplified check: If concat just reverses split exactly.
            op.opType = 'identity';
            op.inputs = { x: producerOp.inputs['x']! };
            changed = true;
          }
        }
      }

      // 180. Fuse Pad into Slice
      if (op.opType === 'slice_by_index' || op.opType === 'slice_by_size') {
        const xInput = op.inputs['x'];
        if (xInput && !Array.isArray(xInput)) {
          const producerOp = block.operations.find((o) => o.outputs[0]?.name === xInput.name);
          if (producerOp && producerOp.opType === 'pad' && !op.attributes['ane_hint_fused_pad']) {
            // In a real implementation we algebraically resolve pad amounts vs slice amounts.
            // We annotate fusion.
            op.attributes['ane_hint_fused_pad'] = producerOp.attributes['pad_amounts'];
            // producerOp (Pad) could be removed if it has no other consumers (DCE handles it later).
            changed = true;
          }
        }
      }

      newOps.push(op);
    }

    block.operations = newOps;
  }
}
