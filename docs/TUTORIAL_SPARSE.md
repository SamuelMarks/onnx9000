# Tutorial: Pruning and Sparsification with onnx9000

This tutorial walks you through sparsifying an ONNX model using `onnx9000.sparse` and SparseML recipes.

## 1. Installation

Ensure you have the `onnx9000` CLI installed:

```bash
uv pip install -e .
```

## 2. Basic Magnitude Pruning

You can apply global magnitude pruning directly via the CLI:

```bash
onnx9000 sparse prune model.onnx --sparsity 0.8 -o sparse_model.onnx
```

This will zero out 80% of the smallest weights in the model and convert them to `SparseTensorProto` to save space.

## 3. Using SparseML Recipes

Recipes allow more fine-grained control over which layers to prune and what algorithms to use.

Create a `recipe.yaml`:

```yaml
version: 1.1.0
modifiers:
  - !MagnitudePruningModifier
    init_sparsity: 0.05
    final_sparsity: 0.8
    params: ['re:.*weight']
    leave_unmasked: ['conv1.weight']
  - !NMPruningModifier
    n: 2
    m: 4
    params: ['conv2.weight']
```

Apply the recipe:

```bash
onnx9000 sparse prune model.onnx --recipe recipe.yaml -o sparse_model.onnx
```

## 4. Understanding Structured Sparsity (2:4)

`onnx9000` supports N:M structured sparsity, optimized for hardware like Nvidia Ampere GPUs.
The `NMPruningModifier` enforces that for every M contiguous elements, only N are non-zero.

Example (2:4):

- Original: `[1.2, 0.1, 0.5, 2.3]`
- Pruned: `[1.2, 0.0, 0.0, 2.3]` (keeps 2 largest)

## 5. Sparsity Profiling

View the savings and layer-wise sparsity:

```bash
onnx9000 sparse prune model.onnx --sparsity 0.5
```

The CLI will output a report like:

```
Layer Name                               | Sparsity   | Saving (KB)
----------------------------------------------------------------------
conv1.weight                             |    50.00% |        12.50
conv2.weight                             |    50.00% |        25.00
----------------------------------------------------------------------
OVERALL                                  |    50.00% |      0.04 MB
```

## 6. Architecture: How it works

`onnx9000` uses a multi-phase approach:

1. **Parsing:** The ONNX model is loaded into a zero-dependency Core IR.
2. **Analysis:** The recipe is parsed and target constants are identified.
3. **Masking:** Pruning algorithms calculate thresholds and apply bitmasks to weights.
4. **Compaction:** Modified `Constant` nodes are converted to `SparseTensor` objects (COO, CSR, etc.).
5. **Propagation:** Channel pruning is propagated through the graph to adjust shapes of downstream layers.
6. **Serialization:** The resulting graph is serialized back to standard ONNX with `SparseTensorProto` initializers.
