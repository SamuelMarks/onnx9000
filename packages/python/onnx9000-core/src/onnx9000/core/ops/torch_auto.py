"""Auto-generated core ops for torch compliance."""

from typing import Any, Optional
from onnx9000.core.ir import Tensor
from onnx9000.core.registry import register_op
from onnx9000.core.ops import record_op


@register_op("Boolstorage", "ai.onnx")
def BoolStorage(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Boolstorage."""
    return record_op("Boolstorage", [x] + list(args), kwargs)


@register_op("Booltensor", "ai.onnx")
def BoolTensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Booltensor."""
    return record_op("Booltensor", [x] + list(args), kwargs)


@register_op("Bytestorage", "ai.onnx")
def ByteStorage(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bytestorage."""
    return record_op("Bytestorage", [x] + list(args), kwargs)


@register_op("Bytetensor", "ai.onnx")
def ByteTensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bytetensor."""
    return record_op("Bytetensor", [x] + list(args), kwargs)


@register_op("Charstorage", "ai.onnx")
def CharStorage(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Charstorage."""
    return record_op("Charstorage", [x] + list(args), kwargs)


@register_op("Chartensor", "ai.onnx")
def CharTensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Chartensor."""
    return record_op("Chartensor", [x] + list(args), kwargs)


@register_op("Doublestorage", "ai.onnx")
def DoubleStorage(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Doublestorage."""
    return record_op("Doublestorage", [x] + list(args), kwargs)


@register_op("Doubletensor", "ai.onnx")
def DoubleTensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Doubletensor."""
    return record_op("Doubletensor", [x] + list(args), kwargs)


@register_op("Floatstorage", "ai.onnx")
def FloatStorage(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Floatstorage."""
    return record_op("Floatstorage", [x] + list(args), kwargs)


@register_op("Floattensor", "ai.onnx")
def FloatTensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Floattensor."""
    return record_op("Floattensor", [x] + list(args), kwargs)


@register_op("Gradscaler", "ai.onnx")
def GradScaler(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Gradscaler."""
    return record_op("Gradscaler", [x] + list(args), kwargs)


@register_op("Intstorage", "ai.onnx")
def IntStorage(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Intstorage."""
    return record_op("Intstorage", [x] + list(args), kwargs)


@register_op("Inttensor", "ai.onnx")
def IntTensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Inttensor."""
    return record_op("Inttensor", [x] + list(args), kwargs)


@register_op("Longstorage", "ai.onnx")
def LongStorage(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Longstorage."""
    return record_op("Longstorage", [x] + list(args), kwargs)


@register_op("Longtensor", "ai.onnx")
def LongTensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Longtensor."""
    return record_op("Longtensor", [x] + list(args), kwargs)


@register_op("Shortstorage", "ai.onnx")
def ShortStorage(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Shortstorage."""
    return record_op("Shortstorage", [x] + list(args), kwargs)


@register_op("Shorttensor", "ai.onnx")
def ShortTensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Shorttensor."""
    return record_op("Shorttensor", [x] + list(args), kwargs)


@register_op("Symbool", "ai.onnx")
def SymBool(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Symbool."""
    return record_op("Symbool", [x] + list(args), kwargs)


@register_op("Symfloat", "ai.onnx")
def SymFloat(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Symfloat."""
    return record_op("Symfloat", [x] + list(args), kwargs)


@register_op("Symint", "ai.onnx")
def SymInt(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Symint."""
    return record_op("Symint", [x] + list(args), kwargs)


@register_op("Typedstorage", "ai.onnx")
def TypedStorage(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Typedstorage."""
    return record_op("Typedstorage", [x] + list(args), kwargs)


@register_op("Untypedstorage", "ai.onnx")
def UntypedStorage(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Untypedstorage."""
    return record_op("Untypedstorage", [x] + list(args), kwargs)


@register_op("Are_deterministic_algorithms_enabled", "ai.onnx")
def are_deterministic_algorithms_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Are_deterministic_algorithms_enabled."""
    return record_op("Are_deterministic_algorithms_enabled", [x] + list(args), kwargs)


@register_op("Autocast", "ai.onnx")
def autocast(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Autocast."""
    return record_op("Autocast", [x] + list(args), kwargs)


@register_op("Chunk", "ai.onnx")
def chunk(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Chunk."""
    return record_op("Chunk", [x] + list(args), kwargs)


@register_op("Compile", "ai.onnx")
def compile(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Compile."""
    return record_op("Compile", [x] + list(args), kwargs)


@register_op("Cond", "ai.onnx")
def cond(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cond."""
    return record_op("Cond", [x] + list(args), kwargs)


@register_op("Enable_grad", "ai.onnx")
def enable_grad(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Enable_grad."""
    return record_op("Enable_grad", [x] + list(args), kwargs)


@register_op("Export_additionalinputs", "ai.onnx")
def export_AdditionalInputs(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_additionalinputs."""
    return record_op("Export_additionalinputs", [x] + list(args), kwargs)


@register_op("Export_constraint", "ai.onnx")
def export_Constraint(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_constraint."""
    return record_op("Export_constraint", [x] + list(args), kwargs)


@register_op("Export_customdecomptable", "ai.onnx")
def export_CustomDecompTable(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_customdecomptable."""
    return record_op("Export_customdecomptable", [x] + list(args), kwargs)


@register_op("Export_default_decompositions", "ai.onnx")
def export_default_decompositions(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_default_decompositions."""
    return record_op("Export_default_decompositions", [x] + list(args), kwargs)


@register_op("Export_dim", "ai.onnx")
def export_Dim(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_dim."""
    return record_op("Export_dim", [x] + list(args), kwargs)


@register_op("Export_dims", "ai.onnx")
def export_dims(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_dims."""
    return record_op("Export_dims", [x] + list(args), kwargs)


@register_op("Export_draft_export", "ai.onnx")
def export_draft_export(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_draft_export."""
    return record_op("Export_draft_export", [x] + list(args), kwargs)


@register_op("Export_export", "ai.onnx")
def export_export(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_export."""
    return record_op("Export_export", [x] + list(args), kwargs)


@register_op("Export_exportbackwardsignature", "ai.onnx")
def export_ExportBackwardSignature(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_exportbackwardsignature."""
    return record_op("Export_exportbackwardsignature", [x] + list(args), kwargs)


@register_op("Export_exportedprogram", "ai.onnx")
def export_ExportedProgram(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_exportedprogram."""
    return record_op("Export_exportedprogram", [x] + list(args), kwargs)


@register_op("Export_exportgraphsignature", "ai.onnx")
def export_ExportGraphSignature(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_exportgraphsignature."""
    return record_op("Export_exportgraphsignature", [x] + list(args), kwargs)


@register_op("Export_flatargsadapter", "ai.onnx")
def export_FlatArgsAdapter(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_flatargsadapter."""
    return record_op("Export_flatargsadapter", [x] + list(args), kwargs)


@register_op("Export_load", "ai.onnx")
def export_load(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_load."""
    return record_op("Export_load", [x] + list(args), kwargs)


@register_op("Export_modulecallentry", "ai.onnx")
def export_ModuleCallEntry(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_modulecallentry."""
    return record_op("Export_modulecallentry", [x] + list(args), kwargs)


@register_op("Export_modulecallsignature", "ai.onnx")
def export_ModuleCallSignature(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_modulecallsignature."""
    return record_op("Export_modulecallsignature", [x] + list(args), kwargs)


@register_op("Export_register_dataclass", "ai.onnx")
def export_register_dataclass(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_register_dataclass."""
    return record_op("Export_register_dataclass", [x] + list(args), kwargs)


@register_op("Export_save", "ai.onnx")
def export_save(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_save."""
    return record_op("Export_save", [x] + list(args), kwargs)


@register_op("Export_shapescollection", "ai.onnx")
def export_ShapesCollection(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_shapescollection."""
    return record_op("Export_shapescollection", [x] + list(args), kwargs)


@register_op("Export_unflatten", "ai.onnx")
def export_unflatten(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_unflatten."""
    return record_op("Export_unflatten", [x] + list(args), kwargs)


@register_op("Export_unflattenedmodule", "ai.onnx")
def export_UnflattenedModule(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_unflattenedmodule."""
    return record_op("Export_unflattenedmodule", [x] + list(args), kwargs)


@register_op("Get_default_device", "ai.onnx")
def get_default_device(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_default_device."""
    return record_op("Get_default_device", [x] + list(args), kwargs)


@register_op("Get_deterministic_debug_mode", "ai.onnx")
def get_deterministic_debug_mode(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_deterministic_debug_mode."""
    return record_op("Get_deterministic_debug_mode", [x] + list(args), kwargs)


@register_op("Get_device_module", "ai.onnx")
def get_device_module(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_device_module."""
    return record_op("Get_device_module", [x] + list(args), kwargs)


@register_op("Get_float32_matmul_precision", "ai.onnx")
def get_float32_matmul_precision(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_float32_matmul_precision."""
    return record_op("Get_float32_matmul_precision", [x] + list(args), kwargs)


@register_op("Get_rng_state", "ai.onnx")
def get_rng_state(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_rng_state."""
    return record_op("Get_rng_state", [x] + list(args), kwargs)


@register_op("Inference_mode", "ai.onnx")
def inference_mode(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Inference_mode."""
    return record_op("Inference_mode", [x] + list(args), kwargs)


@register_op("Initial_seed", "ai.onnx")
def initial_seed(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Initial_seed."""
    return record_op("Initial_seed", [x] + list(args), kwargs)


@register_op("Is_deterministic_algorithms_warn_only_enabled", "ai.onnx")
def is_deterministic_algorithms_warn_only_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_deterministic_algorithms_warn_only_enabled."""
    return record_op("Is_deterministic_algorithms_warn_only_enabled", [x] + list(args), kwargs)


@register_op("Is_storage", "ai.onnx")
def is_storage(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_storage."""
    return record_op("Is_storage", [x] + list(args), kwargs)


@register_op("Is_tensor", "ai.onnx")
def is_tensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_tensor."""
    return record_op("Is_tensor", [x] + list(args), kwargs)


@register_op("Is_warn_always_enabled", "ai.onnx")
def is_warn_always_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_warn_always_enabled."""
    return record_op("Is_warn_always_enabled", [x] + list(args), kwargs)


@register_op("Load", "ai.onnx")
def load(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Load."""
    return record_op("Load", [x] + list(args), kwargs)


@register_op("Lobpcg", "ai.onnx")
def lobpcg(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Lobpcg."""
    return record_op("Lobpcg", [x] + list(args), kwargs)


@register_op("Manual_seed", "ai.onnx")
def manual_seed(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Manual_seed."""
    return record_op("Manual_seed", [x] + list(args), kwargs)


@register_op("No_grad", "ai.onnx")
def no_grad(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute No_grad."""
    return record_op("No_grad", [x] + list(args), kwargs)


@register_op("Rand", "ai.onnx")
def rand(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rand."""
    return record_op("Rand", [x] + list(args), kwargs)


@register_op("Save", "ai.onnx")
def save(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Save."""
    return record_op("Save", [x] + list(args), kwargs)


@register_op("Seed", "ai.onnx")
def seed(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Seed."""
    return record_op("Seed", [x] + list(args), kwargs)


@register_op("Set_default_device", "ai.onnx")
def set_default_device(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_default_device."""
    return record_op("Set_default_device", [x] + list(args), kwargs)


@register_op("Set_default_tensor_type", "ai.onnx")
def set_default_tensor_type(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_default_tensor_type."""
    return record_op("Set_default_tensor_type", [x] + list(args), kwargs)


@register_op("Set_deterministic_debug_mode", "ai.onnx")
def set_deterministic_debug_mode(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_deterministic_debug_mode."""
    return record_op("Set_deterministic_debug_mode", [x] + list(args), kwargs)


@register_op("Set_float32_matmul_precision", "ai.onnx")
def set_float32_matmul_precision(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_float32_matmul_precision."""
    return record_op("Set_float32_matmul_precision", [x] + list(args), kwargs)


@register_op("Set_printoptions", "ai.onnx")
def set_printoptions(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_printoptions."""
    return record_op("Set_printoptions", [x] + list(args), kwargs)


@register_op("Set_rng_state", "ai.onnx")
def set_rng_state(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_rng_state."""
    return record_op("Set_rng_state", [x] + list(args), kwargs)


@register_op("Set_warn_always", "ai.onnx")
def set_warn_always(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_warn_always."""
    return record_op("Set_warn_always", [x] + list(args), kwargs)


@register_op("Split", "ai.onnx")
def split(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Split."""
    return record_op("Split", [x] + list(args), kwargs)


@register_op("Stack", "ai.onnx")
def stack(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Stack."""
    return record_op("Stack", [x] + list(args), kwargs)


@register_op("Sym_float", "ai.onnx")
def sym_float(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sym_float."""
    return record_op("Sym_float", [x] + list(args), kwargs)


@register_op("Sym_fresh_size", "ai.onnx")
def sym_fresh_size(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sym_fresh_size."""
    return record_op("Sym_fresh_size", [x] + list(args), kwargs)


@register_op("Sym_int", "ai.onnx")
def sym_int(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sym_int."""
    return record_op("Sym_int", [x] + list(args), kwargs)


@register_op("Sym_ite", "ai.onnx")
def sym_ite(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sym_ite."""
    return record_op("Sym_ite", [x] + list(args), kwargs)


@register_op("Sym_max", "ai.onnx")
def sym_max(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sym_max."""
    return record_op("Sym_max", [x] + list(args), kwargs)


@register_op("Sym_min", "ai.onnx")
def sym_min(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sym_min."""
    return record_op("Sym_min", [x] + list(args), kwargs)


@register_op("Sym_not", "ai.onnx")
def sym_not(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sym_not."""
    return record_op("Sym_not", [x] + list(args), kwargs)


@register_op("Sym_sum", "ai.onnx")
def sym_sum(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sym_sum."""
    return record_op("Sym_sum", [x] + list(args), kwargs)


@register_op("Typename", "ai.onnx")
def typename(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Typename."""
    return record_op("Typename", [x] + list(args), kwargs)


@register_op("Unravel_index", "ai.onnx")
def unravel_index(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Unravel_index."""
    return record_op("Unravel_index", [x] + list(args), kwargs)


@register_op("Use_deterministic_algorithms", "ai.onnx")
def use_deterministic_algorithms(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Use_deterministic_algorithms."""
    return record_op("Use_deterministic_algorithms", [x] + list(args), kwargs)


@register_op("Vmap", "ai.onnx")
def vmap(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Vmap."""
    return record_op("Vmap", [x] + list(args), kwargs)


@register_op("Sym_sqrt", "ai.onnx")
def sym_sqrt(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sym_sqrt."""
    return record_op("Sym_sqrt", [x] + list(args), kwargs)


@register_op("Avg", "ai.onnx")
def AVG(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Avg."""
    return record_op("Avg", [x] + list(args), kwargs)


@register_op("Acceleratorerror", "ai.onnx")
def AcceleratorError(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Acceleratorerror."""
    return record_op("Acceleratorerror", [x] + list(args), kwargs)


@register_op("Aggregationtype", "ai.onnx")
def AggregationType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Aggregationtype."""
    return record_op("Aggregationtype", [x] + list(args), kwargs)


@register_op("Aliasdb", "ai.onnx")
def AliasDb(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Aliasdb."""
    return record_op("Aliasdb", [x] + list(args), kwargs)


@register_op("Anytype", "ai.onnx")
def AnyType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Anytype."""
    return record_op("Anytype", [x] + list(args), kwargs)


@register_op("Argument", "ai.onnx")
def Argument(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Argument."""
    return record_op("Argument", [x] + list(args), kwargs)


@register_op("Argumentspec", "ai.onnx")
def ArgumentSpec(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Argumentspec."""
    return record_op("Argumentspec", [x] + list(args), kwargs)


@register_op("Awaittype", "ai.onnx")
def AwaitType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Awaittype."""
    return record_op("Awaittype", [x] + list(args), kwargs)


@register_op("Benchmarkconfig", "ai.onnx")
def BenchmarkConfig(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Benchmarkconfig."""
    return record_op("Benchmarkconfig", [x] + list(args), kwargs)


@register_op("Benchmarkexecutionstats", "ai.onnx")
def BenchmarkExecutionStats(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Benchmarkexecutionstats."""
    return record_op("Benchmarkexecutionstats", [x] + list(args), kwargs)


@register_op("Block", "ai.onnx")
def Block(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Block."""
    return record_op("Block", [x] + list(args), kwargs)


@register_op("Booltype", "ai.onnx")
def BoolType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Booltype."""
    return record_op("Booltype", [x] + list(args), kwargs)


@register_op("Bufferdict", "ai.onnx")
def BufferDict(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bufferdict."""
    return record_op("Bufferdict", [x] + list(args), kwargs)


@register_op("Callstack", "ai.onnx")
def CallStack(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Callstack."""
    return record_op("Callstack", [x] + list(args), kwargs)


@register_op("Capsule", "ai.onnx")
def Capsule(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Capsule."""
    return record_op("Capsule", [x] + list(args), kwargs)


@register_op("Classtype", "ai.onnx")
def ClassType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Classtype."""
    return record_op("Classtype", [x] + list(args), kwargs)


@register_op("Code", "ai.onnx")
def Code(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Code."""
    return record_op("Code", [x] + list(args), kwargs)


@register_op("Compilationunit", "ai.onnx")
def CompilationUnit(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Compilationunit."""
    return record_op("Compilationunit", [x] + list(args), kwargs)


@register_op("Completeargumentspec", "ai.onnx")
def CompleteArgumentSpec(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Completeargumentspec."""
    return record_op("Completeargumentspec", [x] + list(args), kwargs)


@register_op("Complextype", "ai.onnx")
def ComplexType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Complextype."""
    return record_op("Complextype", [x] + list(args), kwargs)


@register_op("Concretemoduletype", "ai.onnx")
def ConcreteModuleType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Concretemoduletype."""
    return record_op("Concretemoduletype", [x] + list(args), kwargs)


@register_op("Concretemoduletypebuilder", "ai.onnx")
def ConcreteModuleTypeBuilder(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Concretemoduletypebuilder."""
    return record_op("Concretemoduletypebuilder", [x] + list(args), kwargs)


@register_op("Deepcopymemotable", "ai.onnx")
def DeepCopyMemoTable(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Deepcopymemotable."""
    return record_op("Deepcopymemotable", [x] + list(args), kwargs)


@register_op("Deserializationstoragecontext", "ai.onnx")
def DeserializationStorageContext(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Deserializationstoragecontext."""
    return record_op("Deserializationstoragecontext", [x] + list(args), kwargs)


@register_op("Deviceobjtype", "ai.onnx")
def DeviceObjType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Deviceobjtype."""
    return record_op("Deviceobjtype", [x] + list(args), kwargs)


@register_op("Dicttype", "ai.onnx")
def DictType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Dicttype."""
    return record_op("Dicttype", [x] + list(args), kwargs)


@register_op("Disabletorchfunction", "ai.onnx")
def DisableTorchFunction(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Disabletorchfunction."""
    return record_op("Disabletorchfunction", [x] + list(args), kwargs)


@register_op("Disabletorchfunctionsubclass", "ai.onnx")
def DisableTorchFunctionSubclass(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Disabletorchfunctionsubclass."""
    return record_op("Disabletorchfunctionsubclass", [x] + list(args), kwargs)


@register_op("Dispatchkey", "ai.onnx")
def DispatchKey(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Dispatchkey."""
    return record_op("Dispatchkey", [x] + list(args), kwargs)


@register_op("Dispatchkeyset", "ai.onnx")
def DispatchKeySet(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Dispatchkeyset."""
    return record_op("Dispatchkeyset", [x] + list(args), kwargs)


@register_op("Enumtype", "ai.onnx")
def EnumType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Enumtype."""
    return record_op("Enumtype", [x] + list(args), kwargs)


@register_op("Errorreport", "ai.onnx")
def ErrorReport(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Errorreport."""
    return record_op("Errorreport", [x] + list(args), kwargs)


@register_op("Event", "ai.onnx")
def Event(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Event."""
    return record_op("Event", [x] + list(args), kwargs)


@register_op("Excludedispatchkeyguard", "ai.onnx")
def ExcludeDispatchKeyGuard(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Excludedispatchkeyguard."""
    return record_op("Excludedispatchkeyguard", [x] + list(args), kwargs)


@register_op("Executionplan", "ai.onnx")
def ExecutionPlan(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Executionplan."""
    return record_op("Executionplan", [x] + list(args), kwargs)


@register_op("Fatalerror", "ai.onnx")
def FatalError(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fatalerror."""
    return record_op("Fatalerror", [x] + list(args), kwargs)


@register_op("Filecheck", "ai.onnx")
def FileCheck(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Filecheck."""
    return record_op("Filecheck", [x] + list(args), kwargs)


@register_op("Floattype", "ai.onnx")
def FloatType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Floattype."""
    return record_op("Floattype", [x] + list(args), kwargs)


@register_op("Functionschema", "ai.onnx")
def FunctionSchema(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Functionschema."""
    return record_op("Functionschema", [x] + list(args), kwargs)


@register_op("Future", "ai.onnx")
def Future(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Future."""
    return record_op("Future", [x] + list(args), kwargs)


@register_op("Futuretype", "ai.onnx")
def FutureType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Futuretype."""
    return record_op("Futuretype", [x] + list(args), kwargs)


@register_op("Generator", "ai.onnx")
def Generator(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Generator."""
    return record_op("Generator", [x] + list(args), kwargs)


@register_op("Gradient", "ai.onnx")
def Gradient(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Gradient."""
    return record_op("Gradient", [x] + list(args), kwargs)


@register_op("Graph", "ai.onnx")
def Graph(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Graph."""
    return record_op("Graph", [x] + list(args), kwargs)


@register_op("Graphexecutorstate", "ai.onnx")
def GraphExecutorState(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Graphexecutorstate."""
    return record_op("Graphexecutorstate", [x] + list(args), kwargs)


@register_op("Iodescriptor", "ai.onnx")
def IODescriptor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Iodescriptor."""
    return record_op("Iodescriptor", [x] + list(args), kwargs)


@register_op("Inferredtype", "ai.onnx")
def InferredType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Inferredtype."""
    return record_op("Inferredtype", [x] + list(args), kwargs)


@register_op("Inttype", "ai.onnx")
def IntType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Inttype."""
    return record_op("Inttype", [x] + list(args), kwargs)


@register_op("Interfacetype", "ai.onnx")
def InterfaceType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Interfacetype."""
    return record_op("Interfacetype", [x] + list(args), kwargs)


@register_op("Jitexception", "ai.onnx")
def JITException(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Jitexception."""
    return record_op("Jitexception", [x] + list(args), kwargs)


@register_op("Listtype", "ai.onnx")
def ListType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Listtype."""
    return record_op("Listtype", [x] + list(args), kwargs)


@register_op("Litescriptmodule", "ai.onnx")
def LiteScriptModule(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Litescriptmodule."""
    return record_op("Litescriptmodule", [x] + list(args), kwargs)


@register_op("Lockinglogger", "ai.onnx")
def LockingLogger(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Lockinglogger."""
    return record_op("Lockinglogger", [x] + list(args), kwargs)


@register_op("Moduledict", "ai.onnx")
def ModuleDict(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Moduledict."""
    return record_op("Moduledict", [x] + list(args), kwargs)


@register_op("Node", "ai.onnx")
def Node(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Node."""
    return record_op("Node", [x] + list(args), kwargs)


@register_op("Nonetype", "ai.onnx")
def NoneType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Nonetype."""
    return record_op("Nonetype", [x] + list(args), kwargs)


@register_op("Nooplogger", "ai.onnx")
def NoopLogger(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Nooplogger."""
    return record_op("Nooplogger", [x] + list(args), kwargs)


@register_op("Numbertype", "ai.onnx")
def NumberType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Numbertype."""
    return record_op("Numbertype", [x] + list(args), kwargs)


@register_op("Operatorinfo", "ai.onnx")
def OperatorInfo(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Operatorinfo."""
    return record_op("Operatorinfo", [x] + list(args), kwargs)


@register_op("Optionaltype", "ai.onnx")
def OptionalType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Optionaltype."""
    return record_op("Optionaltype", [x] + list(args), kwargs)


@register_op("Outofmemoryerror", "ai.onnx")
def OutOfMemoryError(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Outofmemoryerror."""
    return record_op("Outofmemoryerror", [x] + list(args), kwargs)


@register_op("Parameterdict", "ai.onnx")
def ParameterDict(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Parameterdict."""
    return record_op("Parameterdict", [x] + list(args), kwargs)


@register_op("Pyobjecttype", "ai.onnx")
def PyObjectType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Pyobjecttype."""
    return record_op("Pyobjecttype", [x] + list(args), kwargs)


@register_op("Pytorchfilereader", "ai.onnx")
def PyTorchFileReader(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Pytorchfilereader."""
    return record_op("Pytorchfilereader", [x] + list(args), kwargs)


@register_op("Pytorchfilewriter", "ai.onnx")
def PyTorchFileWriter(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Pytorchfilewriter."""
    return record_op("Pytorchfilewriter", [x] + list(args), kwargs)


@register_op("Rreftype", "ai.onnx")
def RRefType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rreftype."""
    return record_op("Rreftype", [x] + list(args), kwargs)


@register_op("Sum", "ai.onnx")
def SUM(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sum."""
    return record_op("Sum", [x] + list(args), kwargs)


@register_op("Scriptclass", "ai.onnx")
def ScriptClass(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scriptclass."""
    return record_op("Scriptclass", [x] + list(args), kwargs)


@register_op("Scriptclassfunction", "ai.onnx")
def ScriptClassFunction(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scriptclassfunction."""
    return record_op("Scriptclassfunction", [x] + list(args), kwargs)


@register_op("Scriptdict", "ai.onnx")
def ScriptDict(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scriptdict."""
    return record_op("Scriptdict", [x] + list(args), kwargs)


@register_op("Scriptdictiterator", "ai.onnx")
def ScriptDictIterator(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scriptdictiterator."""
    return record_op("Scriptdictiterator", [x] + list(args), kwargs)


@register_op("Scriptdictkeyiterator", "ai.onnx")
def ScriptDictKeyIterator(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scriptdictkeyiterator."""
    return record_op("Scriptdictkeyiterator", [x] + list(args), kwargs)


@register_op("Scriptfunction", "ai.onnx")
def ScriptFunction(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scriptfunction."""
    return record_op("Scriptfunction", [x] + list(args), kwargs)


@register_op("Scriptlist", "ai.onnx")
def ScriptList(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scriptlist."""
    return record_op("Scriptlist", [x] + list(args), kwargs)


@register_op("Scriptlistiterator", "ai.onnx")
def ScriptListIterator(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scriptlistiterator."""
    return record_op("Scriptlistiterator", [x] + list(args), kwargs)


@register_op("Scriptmethod", "ai.onnx")
def ScriptMethod(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scriptmethod."""
    return record_op("Scriptmethod", [x] + list(args), kwargs)


@register_op("Scriptmodule", "ai.onnx")
def ScriptModule(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scriptmodule."""
    return record_op("Scriptmodule", [x] + list(args), kwargs)


@register_op("Scriptmoduleserializer", "ai.onnx")
def ScriptModuleSerializer(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scriptmoduleserializer."""
    return record_op("Scriptmoduleserializer", [x] + list(args), kwargs)


@register_op("Scriptobject", "ai.onnx")
def ScriptObject(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scriptobject."""
    return record_op("Scriptobject", [x] + list(args), kwargs)


@register_op("Scriptobjectproperty", "ai.onnx")
def ScriptObjectProperty(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scriptobjectproperty."""
    return record_op("Scriptobjectproperty", [x] + list(args), kwargs)


@register_op("Serializationstoragecontext", "ai.onnx")
def SerializationStorageContext(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Serializationstoragecontext."""
    return record_op("Serializationstoragecontext", [x] + list(args), kwargs)


@register_op("Size", "ai.onnx")
def Size(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Size."""
    return record_op("Size", [x] + list(args), kwargs)


@register_op("Staticmodule", "ai.onnx")
def StaticModule(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Staticmodule."""
    return record_op("Staticmodule", [x] + list(args), kwargs)


@register_op("Stream", "ai.onnx")
def Stream(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Stream."""
    return record_op("Stream", [x] + list(args), kwargs)


@register_op("Streamobjtype", "ai.onnx")
def StreamObjType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Streamobjtype."""
    return record_op("Streamobjtype", [x] + list(args), kwargs)


@register_op("Stringtype", "ai.onnx")
def StringType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Stringtype."""
    return record_op("Stringtype", [x] + list(args), kwargs)


@register_op("Symbooltype", "ai.onnx")
def SymBoolType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Symbooltype."""
    return record_op("Symbooltype", [x] + list(args), kwargs)


@register_op("Syminttype", "ai.onnx")
def SymIntType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Syminttype."""
    return record_op("Syminttype", [x] + list(args), kwargs)


@register_op("Tag", "ai.onnx")
def Tag(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Tag."""
    return record_op("Tag", [x] + list(args), kwargs)


@register_op("Tensortype", "ai.onnx")
def TensorType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Tensortype."""
    return record_op("Tensortype", [x] + list(args), kwargs)


@register_op("Throughputbenchmark", "ai.onnx")
def ThroughputBenchmark(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Throughputbenchmark."""
    return record_op("Throughputbenchmark", [x] + list(args), kwargs)


@register_op("Tracingstate", "ai.onnx")
def TracingState(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Tracingstate."""
    return record_op("Tracingstate", [x] + list(args), kwargs)


@register_op("Tupletype", "ai.onnx")
def TupleType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Tupletype."""
    return record_op("Tupletype", [x] + list(args), kwargs)


@register_op("Type", "ai.onnx")
def Type(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Type."""
    return record_op("Type", [x] + list(args), kwargs)


@register_op("Uniontype", "ai.onnx")
def UnionType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Uniontype."""
    return record_op("Uniontype", [x] + list(args), kwargs)


@register_op("Use", "ai.onnx")
def Use(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Use."""
    return record_op("Use", [x] + list(args), kwargs)


@register_op("Value", "ai.onnx")
def Value(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Value."""
    return record_op("Value", [x] + list(args), kwargs)


@register_op("Autocast_decrement_nesting", "ai.onnx")
def autocast_decrement_nesting(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Autocast_decrement_nesting."""
    return record_op("Autocast_decrement_nesting", [x] + list(args), kwargs)


@register_op("Autocast_increment_nesting", "ai.onnx")
def autocast_increment_nesting(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Autocast_increment_nesting."""
    return record_op("Autocast_increment_nesting", [x] + list(args), kwargs)


@register_op("Clear_autocast_cache", "ai.onnx")
def clear_autocast_cache(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Clear_autocast_cache."""
    return record_op("Clear_autocast_cache", [x] + list(args), kwargs)


@register_op("Cpp_orderedmoduledict", "ai.onnx")
def cpp_OrderedModuleDict(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cpp_orderedmoduledict."""
    return record_op("Cpp_orderedmoduledict", [x] + list(args), kwargs)


@register_op("Cpp_orderedtensordict", "ai.onnx")
def cpp_OrderedTensorDict(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cpp_orderedtensordict."""
    return record_op("Cpp_orderedtensordict", [x] + list(args), kwargs)


@register_op("Cpp_nn_module", "ai.onnx")
def cpp_nn_Module(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cpp_nn_module."""
    return record_op("Cpp_nn_module", [x] + list(args), kwargs)


@register_op("Default_generator", "ai.onnx")
def default_generator(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Default_generator."""
    return record_op("Default_generator", [x] + list(args), kwargs)


@register_op("Device", "ai.onnx")
def device(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Device."""
    return record_op("Device", [x] + list(args), kwargs)


@register_op("Dtype", "ai.onnx")
def dtype(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Dtype."""
    return record_op("Dtype", [x] + list(args), kwargs)


@register_op("Finfo", "ai.onnx")
def finfo(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Finfo."""
    return record_op("Finfo", [x] + list(args), kwargs)


@register_op("Fork", "ai.onnx")
def fork(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fork."""
    return record_op("Fork", [x] + list(args), kwargs)


@register_op("Get_autocast_cpu_dtype", "ai.onnx")
def get_autocast_cpu_dtype(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_autocast_cpu_dtype."""
    return record_op("Get_autocast_cpu_dtype", [x] + list(args), kwargs)


@register_op("Get_autocast_dtype", "ai.onnx")
def get_autocast_dtype(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_autocast_dtype."""
    return record_op("Get_autocast_dtype", [x] + list(args), kwargs)


@register_op("Get_autocast_gpu_dtype", "ai.onnx")
def get_autocast_gpu_dtype(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_autocast_gpu_dtype."""
    return record_op("Get_autocast_gpu_dtype", [x] + list(args), kwargs)


@register_op("Get_autocast_ipu_dtype", "ai.onnx")
def get_autocast_ipu_dtype(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_autocast_ipu_dtype."""
    return record_op("Get_autocast_ipu_dtype", [x] + list(args), kwargs)


@register_op("Get_autocast_xla_dtype", "ai.onnx")
def get_autocast_xla_dtype(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_autocast_xla_dtype."""
    return record_op("Get_autocast_xla_dtype", [x] + list(args), kwargs)


@register_op("Get_default_dtype", "ai.onnx")
def get_default_dtype(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_default_dtype."""
    return record_op("Get_default_dtype", [x] + list(args), kwargs)


@register_op("Get_num_interop_threads", "ai.onnx")
def get_num_interop_threads(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_num_interop_threads."""
    return record_op("Get_num_interop_threads", [x] + list(args), kwargs)


@register_op("Get_num_threads", "ai.onnx")
def get_num_threads(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_num_threads."""
    return record_op("Get_num_threads", [x] + list(args), kwargs)


@register_op("Has_lapack", "ai.onnx")
def has_lapack(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Has_lapack."""
    return record_op("Has_lapack", [x] + list(args), kwargs)


@register_op("Has_mkl", "ai.onnx")
def has_mkl(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Has_mkl."""
    return record_op("Has_mkl", [x] + list(args), kwargs)


@register_op("Has_openmp", "ai.onnx")
def has_openmp(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Has_openmp."""
    return record_op("Has_openmp", [x] + list(args), kwargs)


@register_op("Has_spectral", "ai.onnx")
def has_spectral(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Has_spectral."""
    return record_op("Has_spectral", [x] + list(args), kwargs)


@register_op("Iinfo", "ai.onnx")
def iinfo(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Iinfo."""
    return record_op("Iinfo", [x] + list(args), kwargs)


@register_op("Import_ir_module", "ai.onnx")
def import_ir_module(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Import_ir_module."""
    return record_op("Import_ir_module", [x] + list(args), kwargs)


@register_op("Import_ir_module_from_buffer", "ai.onnx")
def import_ir_module_from_buffer(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Import_ir_module_from_buffer."""
    return record_op("Import_ir_module_from_buffer", [x] + list(args), kwargs)


@register_op("Init_num_threads", "ai.onnx")
def init_num_threads(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Init_num_threads."""
    return record_op("Init_num_threads", [x] + list(args), kwargs)


@register_op("Is_anomaly_check_nan_enabled", "ai.onnx")
def is_anomaly_check_nan_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_anomaly_check_nan_enabled."""
    return record_op("Is_anomaly_check_nan_enabled", [x] + list(args), kwargs)


@register_op("Is_anomaly_enabled", "ai.onnx")
def is_anomaly_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_anomaly_enabled."""
    return record_op("Is_anomaly_enabled", [x] + list(args), kwargs)


@register_op("Is_autocast_cache_enabled", "ai.onnx")
def is_autocast_cache_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_autocast_cache_enabled."""
    return record_op("Is_autocast_cache_enabled", [x] + list(args), kwargs)


@register_op("Is_autocast_cpu_enabled", "ai.onnx")
def is_autocast_cpu_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_autocast_cpu_enabled."""
    return record_op("Is_autocast_cpu_enabled", [x] + list(args), kwargs)


@register_op("Is_autocast_enabled", "ai.onnx")
def is_autocast_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_autocast_enabled."""
    return record_op("Is_autocast_enabled", [x] + list(args), kwargs)


@register_op("Is_autocast_ipu_enabled", "ai.onnx")
def is_autocast_ipu_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_autocast_ipu_enabled."""
    return record_op("Is_autocast_ipu_enabled", [x] + list(args), kwargs)


@register_op("Is_autocast_xla_enabled", "ai.onnx")
def is_autocast_xla_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_autocast_xla_enabled."""
    return record_op("Is_autocast_xla_enabled", [x] + list(args), kwargs)


@register_op("Is_grad_enabled", "ai.onnx")
def is_grad_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_grad_enabled."""
    return record_op("Is_grad_enabled", [x] + list(args), kwargs)


@register_op("Is_inference_mode_enabled", "ai.onnx")
def is_inference_mode_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_inference_mode_enabled."""
    return record_op("Is_inference_mode_enabled", [x] + list(args), kwargs)


@register_op("Layout", "ai.onnx")
def layout(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Layout."""
    return record_op("Layout", [x] + list(args), kwargs)


@register_op("Memory_format", "ai.onnx")
def memory_format(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Memory_format."""
    return record_op("Memory_format", [x] + list(args), kwargs)


@register_op("Merge_type_from_type_comment", "ai.onnx")
def merge_type_from_type_comment(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Merge_type_from_type_comment."""
    return record_op("Merge_type_from_type_comment", [x] + list(args), kwargs)


@register_op("Parse_ir", "ai.onnx")
def parse_ir(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Parse_ir."""
    return record_op("Parse_ir", [x] + list(args), kwargs)


@register_op("Parse_schema", "ai.onnx")
def parse_schema(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Parse_schema."""
    return record_op("Parse_schema", [x] + list(args), kwargs)


@register_op("Parse_type_comment", "ai.onnx")
def parse_type_comment(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Parse_type_comment."""
    return record_op("Parse_type_comment", [x] + list(args), kwargs)


@register_op("Qscheme", "ai.onnx")
def qscheme(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Qscheme."""
    return record_op("Qscheme", [x] + list(args), kwargs)


@register_op("Read_vitals", "ai.onnx")
def read_vitals(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Read_vitals."""
    return record_op("Read_vitals", [x] + list(args), kwargs)


@register_op("Set_anomaly_enabled", "ai.onnx")
def set_anomaly_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_anomaly_enabled."""
    return record_op("Set_anomaly_enabled", [x] + list(args), kwargs)


@register_op("Set_autocast_cache_enabled", "ai.onnx")
def set_autocast_cache_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_autocast_cache_enabled."""
    return record_op("Set_autocast_cache_enabled", [x] + list(args), kwargs)


@register_op("Set_autocast_cpu_dtype", "ai.onnx")
def set_autocast_cpu_dtype(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_autocast_cpu_dtype."""
    return record_op("Set_autocast_cpu_dtype", [x] + list(args), kwargs)


@register_op("Set_autocast_cpu_enabled", "ai.onnx")
def set_autocast_cpu_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_autocast_cpu_enabled."""
    return record_op("Set_autocast_cpu_enabled", [x] + list(args), kwargs)


@register_op("Set_autocast_dtype", "ai.onnx")
def set_autocast_dtype(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_autocast_dtype."""
    return record_op("Set_autocast_dtype", [x] + list(args), kwargs)


@register_op("Set_autocast_enabled", "ai.onnx")
def set_autocast_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_autocast_enabled."""
    return record_op("Set_autocast_enabled", [x] + list(args), kwargs)


@register_op("Set_autocast_gpu_dtype", "ai.onnx")
def set_autocast_gpu_dtype(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_autocast_gpu_dtype."""
    return record_op("Set_autocast_gpu_dtype", [x] + list(args), kwargs)


@register_op("Set_autocast_ipu_dtype", "ai.onnx")
def set_autocast_ipu_dtype(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_autocast_ipu_dtype."""
    return record_op("Set_autocast_ipu_dtype", [x] + list(args), kwargs)


@register_op("Set_autocast_ipu_enabled", "ai.onnx")
def set_autocast_ipu_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_autocast_ipu_enabled."""
    return record_op("Set_autocast_ipu_enabled", [x] + list(args), kwargs)


@register_op("Set_autocast_xla_dtype", "ai.onnx")
def set_autocast_xla_dtype(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_autocast_xla_dtype."""
    return record_op("Set_autocast_xla_dtype", [x] + list(args), kwargs)


@register_op("Set_autocast_xla_enabled", "ai.onnx")
def set_autocast_xla_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_autocast_xla_enabled."""
    return record_op("Set_autocast_xla_enabled", [x] + list(args), kwargs)


@register_op("Set_flush_denormal", "ai.onnx")
def set_flush_denormal(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_flush_denormal."""
    return record_op("Set_flush_denormal", [x] + list(args), kwargs)


@register_op("Set_num_interop_threads", "ai.onnx")
def set_num_interop_threads(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_num_interop_threads."""
    return record_op("Set_num_interop_threads", [x] + list(args), kwargs)


@register_op("Set_num_threads", "ai.onnx")
def set_num_threads(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_num_threads."""
    return record_op("Set_num_threads", [x] + list(args), kwargs)


@register_op("Set_vital", "ai.onnx")
def set_vital(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_vital."""
    return record_op("Set_vital", [x] + list(args), kwargs)


@register_op("Unify_type_list", "ai.onnx")
def unify_type_list(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Unify_type_list."""
    return record_op("Unify_type_list", [x] + list(args), kwargs)


@register_op("Vitals_enabled", "ai.onnx")
def vitals_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Vitals_enabled."""
    return record_op("Vitals_enabled", [x] + list(args), kwargs)


@register_op("Wait", "ai.onnx")
def wait(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Wait."""
    return record_op("Wait", [x] + list(args), kwargs)


@register_op("E", "ai.onnx")
def e(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute E."""
    return record_op("E", [x] + list(args), kwargs)


@register_op("Pi", "ai.onnx")
def pi(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Pi."""
    return record_op("Pi", [x] + list(args), kwargs)


@register_op("Nan", "ai.onnx")
def nan(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Nan."""
    return record_op("Nan", [x] + list(args), kwargs)


@register_op("Inf", "ai.onnx")
def inf(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Inf."""
    return record_op("Inf", [x] + list(args), kwargs)


@register_op("Newaxis", "ai.onnx")
def newaxis(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Newaxis."""
    return record_op("Newaxis", [x] + list(args), kwargs)


@register_op("Abs_", "ai.onnx")
def abs_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Abs_."""
    return record_op("Abs_", [x] + list(args), kwargs)


@register_op("Absolute", "ai.onnx")
def absolute(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Absolute."""
    return record_op("Absolute", [x] + list(args), kwargs)


@register_op("Acos_", "ai.onnx")
def acos_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Acos_."""
    return record_op("Acos_", [x] + list(args), kwargs)


@register_op("Acosh_", "ai.onnx")
def acosh_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Acosh_."""
    return record_op("Acosh_", [x] + list(args), kwargs)


@register_op("Adaptive_avg_pool1d", "ai.onnx")
def adaptive_avg_pool1d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Adaptive_avg_pool1d."""
    return record_op("Adaptive_avg_pool1d", [x] + list(args), kwargs)


@register_op("Adaptive_max_pool1d", "ai.onnx")
def adaptive_max_pool1d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Adaptive_max_pool1d."""
    return record_op("Adaptive_max_pool1d", [x] + list(args), kwargs)


@register_op("Addbmm", "ai.onnx")
def addbmm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Addbmm."""
    return record_op("Addbmm", [x] + list(args), kwargs)


@register_op("Addcdiv", "ai.onnx")
def addcdiv(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Addcdiv."""
    return record_op("Addcdiv", [x] + list(args), kwargs)


@register_op("Addcmul", "ai.onnx")
def addcmul(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Addcmul."""
    return record_op("Addcmul", [x] + list(args), kwargs)


@register_op("Addmm", "ai.onnx")
def addmm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Addmm."""
    return record_op("Addmm", [x] + list(args), kwargs)


@register_op("Addmv", "ai.onnx")
def addmv(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Addmv."""
    return record_op("Addmv", [x] + list(args), kwargs)


@register_op("Addmv_", "ai.onnx")
def addmv_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Addmv_."""
    return record_op("Addmv_", [x] + list(args), kwargs)


@register_op("Addr", "ai.onnx")
def addr(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Addr."""
    return record_op("Addr", [x] + list(args), kwargs)


@register_op("Adjoint", "ai.onnx")
def adjoint(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Adjoint."""
    return record_op("Adjoint", [x] + list(args), kwargs)


@register_op("Affine_grid_generator", "ai.onnx")
def affine_grid_generator(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Affine_grid_generator."""
    return record_op("Affine_grid_generator", [x] + list(args), kwargs)


@register_op("Alias_copy", "ai.onnx")
def alias_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Alias_copy."""
    return record_op("Alias_copy", [x] + list(args), kwargs)


@register_op("Align_tensors", "ai.onnx")
def align_tensors(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Align_tensors."""
    return record_op("Align_tensors", [x] + list(args), kwargs)


@register_op("All", "ai.onnx")
def all(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute All."""
    return record_op("All", [x] + list(args), kwargs)


@register_op("Allclose", "ai.onnx")
def allclose(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Allclose."""
    return record_op("Allclose", [x] + list(args), kwargs)


@register_op("Alpha_dropout", "ai.onnx")
def alpha_dropout(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Alpha_dropout."""
    return record_op("Alpha_dropout", [x] + list(args), kwargs)


@register_op("Alpha_dropout_", "ai.onnx")
def alpha_dropout_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Alpha_dropout_."""
    return record_op("Alpha_dropout_", [x] + list(args), kwargs)


@register_op("Amax", "ai.onnx")
def amax(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Amax."""
    return record_op("Amax", [x] + list(args), kwargs)


@register_op("Amin", "ai.onnx")
def amin(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Amin."""
    return record_op("Amin", [x] + list(args), kwargs)


@register_op("Aminmax", "ai.onnx")
def aminmax(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Aminmax."""
    return record_op("Aminmax", [x] + list(args), kwargs)


@register_op("Angle", "ai.onnx")
def angle(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Angle."""
    return record_op("Angle", [x] + list(args), kwargs)


@register_op("Any", "ai.onnx")
def any(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Any."""
    return record_op("Any", [x] + list(args), kwargs)


@register_op("Arange", "ai.onnx")
def arange(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arange."""
    return record_op("Arange", [x] + list(args), kwargs)


@register_op("Arccos", "ai.onnx")
def arccos(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arccos."""
    return record_op("Arccos", [x] + list(args), kwargs)


@register_op("Arccos_", "ai.onnx")
def arccos_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arccos_."""
    return record_op("Arccos_", [x] + list(args), kwargs)


@register_op("Arccosh", "ai.onnx")
def arccosh(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arccosh."""
    return record_op("Arccosh", [x] + list(args), kwargs)


@register_op("Arccosh_", "ai.onnx")
def arccosh_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arccosh_."""
    return record_op("Arccosh_", [x] + list(args), kwargs)


@register_op("Arcsin", "ai.onnx")
def arcsin(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arcsin."""
    return record_op("Arcsin", [x] + list(args), kwargs)


@register_op("Arcsin_", "ai.onnx")
def arcsin_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arcsin_."""
    return record_op("Arcsin_", [x] + list(args), kwargs)


@register_op("Arcsinh", "ai.onnx")
def arcsinh(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arcsinh."""
    return record_op("Arcsinh", [x] + list(args), kwargs)


@register_op("Arcsinh_", "ai.onnx")
def arcsinh_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arcsinh_."""
    return record_op("Arcsinh_", [x] + list(args), kwargs)


@register_op("Arctan", "ai.onnx")
def arctan(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arctan."""
    return record_op("Arctan", [x] + list(args), kwargs)


@register_op("Arctan2", "ai.onnx")
def arctan2(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arctan2."""
    return record_op("Arctan2", [x] + list(args), kwargs)


@register_op("Arctan_", "ai.onnx")
def arctan_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arctan_."""
    return record_op("Arctan_", [x] + list(args), kwargs)


@register_op("Arctanh", "ai.onnx")
def arctanh(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arctanh."""
    return record_op("Arctanh", [x] + list(args), kwargs)


@register_op("Arctanh_", "ai.onnx")
def arctanh_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arctanh_."""
    return record_op("Arctanh_", [x] + list(args), kwargs)


@register_op("Argsort", "ai.onnx")
def argsort(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Argsort."""
    return record_op("Argsort", [x] + list(args), kwargs)


@register_op("Argwhere", "ai.onnx")
def argwhere(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Argwhere."""
    return record_op("Argwhere", [x] + list(args), kwargs)


@register_op("As_strided", "ai.onnx")
def as_strided(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute As_strided."""
    return record_op("As_strided", [x] + list(args), kwargs)


@register_op("As_strided_", "ai.onnx")
def as_strided_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute As_strided_."""
    return record_op("As_strided_", [x] + list(args), kwargs)


@register_op("As_strided_copy", "ai.onnx")
def as_strided_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute As_strided_copy."""
    return record_op("As_strided_copy", [x] + list(args), kwargs)


@register_op("As_strided_scatter", "ai.onnx")
def as_strided_scatter(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute As_strided_scatter."""
    return record_op("As_strided_scatter", [x] + list(args), kwargs)


@register_op("As_tensor", "ai.onnx")
def as_tensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute As_tensor."""
    return record_op("As_tensor", [x] + list(args), kwargs)


@register_op("Asarray", "ai.onnx")
def asarray(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Asarray."""
    return record_op("Asarray", [x] + list(args), kwargs)


@register_op("Asin_", "ai.onnx")
def asin_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Asin_."""
    return record_op("Asin_", [x] + list(args), kwargs)


@register_op("Asinh_", "ai.onnx")
def asinh_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Asinh_."""
    return record_op("Asinh_", [x] + list(args), kwargs)


@register_op("Atan2", "ai.onnx")
def atan2(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Atan2."""
    return record_op("Atan2", [x] + list(args), kwargs)


@register_op("Atan_", "ai.onnx")
def atan_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Atan_."""
    return record_op("Atan_", [x] + list(args), kwargs)


@register_op("Atanh_", "ai.onnx")
def atanh_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Atanh_."""
    return record_op("Atanh_", [x] + list(args), kwargs)


@register_op("Atleast_1d", "ai.onnx")
def atleast_1d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Atleast_1d."""
    return record_op("Atleast_1d", [x] + list(args), kwargs)


@register_op("Atleast_2d", "ai.onnx")
def atleast_2d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Atleast_2d."""
    return record_op("Atleast_2d", [x] + list(args), kwargs)


@register_op("Atleast_3d", "ai.onnx")
def atleast_3d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Atleast_3d."""
    return record_op("Atleast_3d", [x] + list(args), kwargs)


@register_op("Avg_pool1d", "ai.onnx")
def avg_pool1d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Avg_pool1d."""
    return record_op("Avg_pool1d", [x] + list(args), kwargs)


@register_op("Baddbmm", "ai.onnx")
def baddbmm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Baddbmm."""
    return record_op("Baddbmm", [x] + list(args), kwargs)


@register_op("Bartlett_window", "ai.onnx")
def bartlett_window(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bartlett_window."""
    return record_op("Bartlett_window", [x] + list(args), kwargs)


@register_op("Batch_norm", "ai.onnx")
def batch_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Batch_norm."""
    return record_op("Batch_norm", [x] + list(args), kwargs)


@register_op("Batch_norm_backward_elemt", "ai.onnx")
def batch_norm_backward_elemt(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Batch_norm_backward_elemt."""
    return record_op("Batch_norm_backward_elemt", [x] + list(args), kwargs)


@register_op("Batch_norm_backward_reduce", "ai.onnx")
def batch_norm_backward_reduce(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Batch_norm_backward_reduce."""
    return record_op("Batch_norm_backward_reduce", [x] + list(args), kwargs)


@register_op("Batch_norm_elemt", "ai.onnx")
def batch_norm_elemt(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Batch_norm_elemt."""
    return record_op("Batch_norm_elemt", [x] + list(args), kwargs)


@register_op("Batch_norm_gather_stats", "ai.onnx")
def batch_norm_gather_stats(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Batch_norm_gather_stats."""
    return record_op("Batch_norm_gather_stats", [x] + list(args), kwargs)


@register_op("Batch_norm_gather_stats_with_counts", "ai.onnx")
def batch_norm_gather_stats_with_counts(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Batch_norm_gather_stats_with_counts."""
    return record_op("Batch_norm_gather_stats_with_counts", [x] + list(args), kwargs)


@register_op("Batch_norm_stats", "ai.onnx")
def batch_norm_stats(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Batch_norm_stats."""
    return record_op("Batch_norm_stats", [x] + list(args), kwargs)


@register_op("Batch_norm_update_stats", "ai.onnx")
def batch_norm_update_stats(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Batch_norm_update_stats."""
    return record_op("Batch_norm_update_stats", [x] + list(args), kwargs)


@register_op("Bilinear", "ai.onnx")
def bilinear(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bilinear."""
    return record_op("Bilinear", [x] + list(args), kwargs)


@register_op("Binary_cross_entropy_with_logits", "ai.onnx")
def binary_cross_entropy_with_logits(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Binary_cross_entropy_with_logits."""
    return record_op("Binary_cross_entropy_with_logits", [x] + list(args), kwargs)


@register_op("Bincount", "ai.onnx")
def bincount(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bincount."""
    return record_op("Bincount", [x] + list(args), kwargs)


@register_op("Binomial", "ai.onnx")
def binomial(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Binomial."""
    return record_op("Binomial", [x] + list(args), kwargs)


@register_op("Bitwise_left_shift", "ai.onnx")
def bitwise_left_shift(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bitwise_left_shift."""
    return record_op("Bitwise_left_shift", [x] + list(args), kwargs)


@register_op("Bitwise_right_shift", "ai.onnx")
def bitwise_right_shift(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bitwise_right_shift."""
    return record_op("Bitwise_right_shift", [x] + list(args), kwargs)


@register_op("Block_diag", "ai.onnx")
def block_diag(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Block_diag."""
    return record_op("Block_diag", [x] + list(args), kwargs)


@register_op("Bmm", "ai.onnx")
def bmm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bmm."""
    return record_op("Bmm", [x] + list(args), kwargs)


@register_op("Broadcast_tensors", "ai.onnx")
def broadcast_tensors(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Broadcast_tensors."""
    return record_op("Broadcast_tensors", [x] + list(args), kwargs)


@register_op("Broadcast_to", "ai.onnx")
def broadcast_to(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Broadcast_to."""
    return record_op("Broadcast_to", [x] + list(args), kwargs)


@register_op("Bucketize", "ai.onnx")
def bucketize(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bucketize."""
    return record_op("Bucketize", [x] + list(args), kwargs)


@register_op("Can_cast", "ai.onnx")
def can_cast(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Can_cast."""
    return record_op("Can_cast", [x] + list(args), kwargs)


@register_op("Cartesian_prod", "ai.onnx")
def cartesian_prod(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cartesian_prod."""
    return record_op("Cartesian_prod", [x] + list(args), kwargs)


@register_op("Cat", "ai.onnx")
def cat(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cat."""
    return record_op("Cat", [x] + list(args), kwargs)


@register_op("Ccol_indices_copy", "ai.onnx")
def ccol_indices_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Ccol_indices_copy."""
    return record_op("Ccol_indices_copy", [x] + list(args), kwargs)


@register_op("Cdist", "ai.onnx")
def cdist(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cdist."""
    return record_op("Cdist", [x] + list(args), kwargs)


@register_op("Ceil_", "ai.onnx")
def ceil_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Ceil_."""
    return record_op("Ceil_", [x] + list(args), kwargs)


@register_op("Celu_", "ai.onnx")
def celu_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Celu_."""
    return record_op("Celu_", [x] + list(args), kwargs)


@register_op("Chain_matmul", "ai.onnx")
def chain_matmul(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Chain_matmul."""
    return record_op("Chain_matmul", [x] + list(args), kwargs)


@register_op("Channel_shuffle", "ai.onnx")
def channel_shuffle(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Channel_shuffle."""
    return record_op("Channel_shuffle", [x] + list(args), kwargs)


@register_op("Cholesky", "ai.onnx")
def cholesky(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cholesky."""
    return record_op("Cholesky", [x] + list(args), kwargs)


@register_op("Cholesky_inverse", "ai.onnx")
def cholesky_inverse(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cholesky_inverse."""
    return record_op("Cholesky_inverse", [x] + list(args), kwargs)


@register_op("Cholesky_solve", "ai.onnx")
def cholesky_solve(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cholesky_solve."""
    return record_op("Cholesky_solve", [x] + list(args), kwargs)


@register_op("Choose_qparams_optimized", "ai.onnx")
def choose_qparams_optimized(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Choose_qparams_optimized."""
    return record_op("Choose_qparams_optimized", [x] + list(args), kwargs)


@register_op("Clamp", "ai.onnx")
def clamp(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Clamp."""
    return record_op("Clamp", [x] + list(args), kwargs)


@register_op("Clamp_", "ai.onnx")
def clamp_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Clamp_."""
    return record_op("Clamp_", [x] + list(args), kwargs)


@register_op("Clamp_max", "ai.onnx")
def clamp_max(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Clamp_max."""
    return record_op("Clamp_max", [x] + list(args), kwargs)


@register_op("Clamp_max_", "ai.onnx")
def clamp_max_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Clamp_max_."""
    return record_op("Clamp_max_", [x] + list(args), kwargs)


@register_op("Clamp_min", "ai.onnx")
def clamp_min(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Clamp_min."""
    return record_op("Clamp_min", [x] + list(args), kwargs)


@register_op("Clamp_min_", "ai.onnx")
def clamp_min_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Clamp_min_."""
    return record_op("Clamp_min_", [x] + list(args), kwargs)


@register_op("Clip_", "ai.onnx")
def clip_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Clip_."""
    return record_op("Clip_", [x] + list(args), kwargs)


@register_op("Clone", "ai.onnx")
def clone(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Clone."""
    return record_op("Clone", [x] + list(args), kwargs)


@register_op("Col_indices_copy", "ai.onnx")
def col_indices_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Col_indices_copy."""
    return record_op("Col_indices_copy", [x] + list(args), kwargs)


@register_op("Column_stack", "ai.onnx")
def column_stack(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Column_stack."""
    return record_op("Column_stack", [x] + list(args), kwargs)


@register_op("Combinations", "ai.onnx")
def combinations(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Combinations."""
    return record_op("Combinations", [x] + list(args), kwargs)


@register_op("Complex", "ai.onnx")
def complex(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Complex."""
    return record_op("Complex", [x] + list(args), kwargs)


@register_op("Concatenate", "ai.onnx")
def concatenate(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Concatenate."""
    return record_op("Concatenate", [x] + list(args), kwargs)


@register_op("Conj", "ai.onnx")
def conj(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Conj."""
    return record_op("Conj", [x] + list(args), kwargs)


@register_op("Conj_physical", "ai.onnx")
def conj_physical(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Conj_physical."""
    return record_op("Conj_physical", [x] + list(args), kwargs)


@register_op("Conj_physical_", "ai.onnx")
def conj_physical_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Conj_physical_."""
    return record_op("Conj_physical_", [x] + list(args), kwargs)


@register_op("Constant_pad_nd", "ai.onnx")
def constant_pad_nd(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Constant_pad_nd."""
    return record_op("Constant_pad_nd", [x] + list(args), kwargs)


@register_op("Conv1d", "ai.onnx")
def conv1d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Conv1d."""
    return record_op("Conv1d", [x] + list(args), kwargs)


@register_op("Conv2d", "ai.onnx")
def conv2d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Conv2d."""
    return record_op("Conv2d", [x] + list(args), kwargs)


@register_op("Conv3d", "ai.onnx")
def conv3d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Conv3d."""
    return record_op("Conv3d", [x] + list(args), kwargs)


@register_op("Conv_tbc", "ai.onnx")
def conv_tbc(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Conv_tbc."""
    return record_op("Conv_tbc", [x] + list(args), kwargs)


@register_op("Conv_transpose1d", "ai.onnx")
def conv_transpose1d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Conv_transpose1d."""
    return record_op("Conv_transpose1d", [x] + list(args), kwargs)


@register_op("Conv_transpose2d", "ai.onnx")
def conv_transpose2d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Conv_transpose2d."""
    return record_op("Conv_transpose2d", [x] + list(args), kwargs)


@register_op("Conv_transpose3d", "ai.onnx")
def conv_transpose3d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Conv_transpose3d."""
    return record_op("Conv_transpose3d", [x] + list(args), kwargs)


@register_op("Convolution", "ai.onnx")
def convolution(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Convolution."""
    return record_op("Convolution", [x] + list(args), kwargs)


@register_op("Copysign", "ai.onnx")
def copysign(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Copysign."""
    return record_op("Copysign", [x] + list(args), kwargs)


@register_op("Corrcoef", "ai.onnx")
def corrcoef(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Corrcoef."""
    return record_op("Corrcoef", [x] + list(args), kwargs)


@register_op("Cos_", "ai.onnx")
def cos_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cos_."""
    return record_op("Cos_", [x] + list(args), kwargs)


@register_op("Cosh_", "ai.onnx")
def cosh_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cosh_."""
    return record_op("Cosh_", [x] + list(args), kwargs)


@register_op("Cosine_embedding_loss", "ai.onnx")
def cosine_embedding_loss(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cosine_embedding_loss."""
    return record_op("Cosine_embedding_loss", [x] + list(args), kwargs)


@register_op("Cosine_similarity", "ai.onnx")
def cosine_similarity(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cosine_similarity."""
    return record_op("Cosine_similarity", [x] + list(args), kwargs)


@register_op("Count_nonzero", "ai.onnx")
def count_nonzero(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Count_nonzero."""
    return record_op("Count_nonzero", [x] + list(args), kwargs)


@register_op("Cov", "ai.onnx")
def cov(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cov."""
    return record_op("Cov", [x] + list(args), kwargs)


@register_op("Cross", "ai.onnx")
def cross(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cross."""
    return record_op("Cross", [x] + list(args), kwargs)


@register_op("Crow_indices_copy", "ai.onnx")
def crow_indices_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Crow_indices_copy."""
    return record_op("Crow_indices_copy", [x] + list(args), kwargs)


@register_op("Ctc_loss", "ai.onnx")
def ctc_loss(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Ctc_loss."""
    return record_op("Ctc_loss", [x] + list(args), kwargs)


@register_op("Cudnn_affine_grid_generator", "ai.onnx")
def cudnn_affine_grid_generator(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cudnn_affine_grid_generator."""
    return record_op("Cudnn_affine_grid_generator", [x] + list(args), kwargs)


@register_op("Cudnn_batch_norm", "ai.onnx")
def cudnn_batch_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cudnn_batch_norm."""
    return record_op("Cudnn_batch_norm", [x] + list(args), kwargs)


@register_op("Cudnn_convolution", "ai.onnx")
def cudnn_convolution(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cudnn_convolution."""
    return record_op("Cudnn_convolution", [x] + list(args), kwargs)


@register_op("Cudnn_convolution_add_relu", "ai.onnx")
def cudnn_convolution_add_relu(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cudnn_convolution_add_relu."""
    return record_op("Cudnn_convolution_add_relu", [x] + list(args), kwargs)


@register_op("Cudnn_convolution_relu", "ai.onnx")
def cudnn_convolution_relu(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cudnn_convolution_relu."""
    return record_op("Cudnn_convolution_relu", [x] + list(args), kwargs)


@register_op("Cudnn_convolution_transpose", "ai.onnx")
def cudnn_convolution_transpose(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cudnn_convolution_transpose."""
    return record_op("Cudnn_convolution_transpose", [x] + list(args), kwargs)


@register_op("Cudnn_grid_sampler", "ai.onnx")
def cudnn_grid_sampler(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cudnn_grid_sampler."""
    return record_op("Cudnn_grid_sampler", [x] + list(args), kwargs)


@register_op("Cudnn_is_acceptable", "ai.onnx")
def cudnn_is_acceptable(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cudnn_is_acceptable."""
    return record_op("Cudnn_is_acceptable", [x] + list(args), kwargs)


@register_op("Cummax", "ai.onnx")
def cummax(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cummax."""
    return record_op("Cummax", [x] + list(args), kwargs)


@register_op("Cummin", "ai.onnx")
def cummin(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cummin."""
    return record_op("Cummin", [x] + list(args), kwargs)


@register_op("Cumprod", "ai.onnx")
def cumprod(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cumprod."""
    return record_op("Cumprod", [x] + list(args), kwargs)


@register_op("Cumulative_trapezoid", "ai.onnx")
def cumulative_trapezoid(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cumulative_trapezoid."""
    return record_op("Cumulative_trapezoid", [x] + list(args), kwargs)


@register_op("Deg2rad", "ai.onnx")
def deg2rad(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Deg2rad."""
    return record_op("Deg2rad", [x] + list(args), kwargs)


@register_op("Deg2rad_", "ai.onnx")
def deg2rad_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Deg2rad_."""
    return record_op("Deg2rad_", [x] + list(args), kwargs)


@register_op("Dequantize", "ai.onnx")
def dequantize(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Dequantize."""
    return record_op("Dequantize", [x] + list(args), kwargs)


@register_op("Detach", "ai.onnx")
def detach(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Detach."""
    return record_op("Detach", [x] + list(args), kwargs)


@register_op("Detach_", "ai.onnx")
def detach_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Detach_."""
    return record_op("Detach_", [x] + list(args), kwargs)


@register_op("Detach_copy", "ai.onnx")
def detach_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Detach_copy."""
    return record_op("Detach_copy", [x] + list(args), kwargs)


@register_op("Diag", "ai.onnx")
def diag(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Diag."""
    return record_op("Diag", [x] + list(args), kwargs)


@register_op("Diag_embed", "ai.onnx")
def diag_embed(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Diag_embed."""
    return record_op("Diag_embed", [x] + list(args), kwargs)


@register_op("Diagflat", "ai.onnx")
def diagflat(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Diagflat."""
    return record_op("Diagflat", [x] + list(args), kwargs)


@register_op("Diagonal", "ai.onnx")
def diagonal(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Diagonal."""
    return record_op("Diagonal", [x] + list(args), kwargs)


@register_op("Diagonal_copy", "ai.onnx")
def diagonal_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Diagonal_copy."""
    return record_op("Diagonal_copy", [x] + list(args), kwargs)


@register_op("Diagonal_scatter", "ai.onnx")
def diagonal_scatter(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Diagonal_scatter."""
    return record_op("Diagonal_scatter", [x] + list(args), kwargs)


@register_op("Diff", "ai.onnx")
def diff(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Diff."""
    return record_op("Diff", [x] + list(args), kwargs)


@register_op("Digamma", "ai.onnx")
def digamma(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Digamma."""
    return record_op("Digamma", [x] + list(args), kwargs)


@register_op("Dist", "ai.onnx")
def dist(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Dist."""
    return record_op("Dist", [x] + list(args), kwargs)


@register_op("Divide", "ai.onnx")
def divide(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Divide."""
    return record_op("Divide", [x] + list(args), kwargs)


@register_op("Dot", "ai.onnx")
def dot(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Dot."""
    return record_op("Dot", [x] + list(args), kwargs)


@register_op("Dropout_", "ai.onnx")
def dropout_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Dropout_."""
    return record_op("Dropout_", [x] + list(args), kwargs)


@register_op("Dsmm", "ai.onnx")
def dsmm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Dsmm."""
    return record_op("Dsmm", [x] + list(args), kwargs)


@register_op("Dsplit", "ai.onnx")
def dsplit(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Dsplit."""
    return record_op("Dsplit", [x] + list(args), kwargs)


@register_op("Dstack", "ai.onnx")
def dstack(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Dstack."""
    return record_op("Dstack", [x] + list(args), kwargs)


@register_op("Embedding", "ai.onnx")
def embedding(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Embedding."""
    return record_op("Embedding", [x] + list(args), kwargs)


@register_op("Embedding_bag", "ai.onnx")
def embedding_bag(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Embedding_bag."""
    return record_op("Embedding_bag", [x] + list(args), kwargs)


@register_op("Embedding_renorm_", "ai.onnx")
def embedding_renorm_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Embedding_renorm_."""
    return record_op("Embedding_renorm_", [x] + list(args), kwargs)


@register_op("Empty", "ai.onnx")
def empty(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Empty."""
    return record_op("Empty", [x] + list(args), kwargs)


@register_op("Empty_like", "ai.onnx")
def empty_like(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Empty_like."""
    return record_op("Empty_like", [x] + list(args), kwargs)


@register_op("Empty_permuted", "ai.onnx")
def empty_permuted(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Empty_permuted."""
    return record_op("Empty_permuted", [x] + list(args), kwargs)


@register_op("Empty_quantized", "ai.onnx")
def empty_quantized(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Empty_quantized."""
    return record_op("Empty_quantized", [x] + list(args), kwargs)


@register_op("Empty_strided", "ai.onnx")
def empty_strided(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Empty_strided."""
    return record_op("Empty_strided", [x] + list(args), kwargs)


@register_op("Eq", "ai.onnx")
def eq(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Eq."""
    return record_op("Eq", [x] + list(args), kwargs)


@register_op("Erf_", "ai.onnx")
def erf_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Erf_."""
    return record_op("Erf_", [x] + list(args), kwargs)


@register_op("Erfc", "ai.onnx")
def erfc(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Erfc."""
    return record_op("Erfc", [x] + list(args), kwargs)


@register_op("Erfc_", "ai.onnx")
def erfc_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Erfc_."""
    return record_op("Erfc_", [x] + list(args), kwargs)


@register_op("Erfinv", "ai.onnx")
def erfinv(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Erfinv."""
    return record_op("Erfinv", [x] + list(args), kwargs)


@register_op("Exp2", "ai.onnx")
def exp2(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Exp2."""
    return record_op("Exp2", [x] + list(args), kwargs)


@register_op("Exp2_", "ai.onnx")
def exp2_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Exp2_."""
    return record_op("Exp2_", [x] + list(args), kwargs)


@register_op("Exp_", "ai.onnx")
def exp_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Exp_."""
    return record_op("Exp_", [x] + list(args), kwargs)


@register_op("Expand_copy", "ai.onnx")
def expand_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Expand_copy."""
    return record_op("Expand_copy", [x] + list(args), kwargs)


@register_op("Expm1", "ai.onnx")
def expm1(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Expm1."""
    return record_op("Expm1", [x] + list(args), kwargs)


@register_op("Expm1_", "ai.onnx")
def expm1_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Expm1_."""
    return record_op("Expm1_", [x] + list(args), kwargs)


@register_op("Eye", "ai.onnx")
def eye(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Eye."""
    return record_op("Eye", [x] + list(args), kwargs)


@register_op("Fake_quantize_per_channel_affine", "ai.onnx")
def fake_quantize_per_channel_affine(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fake_quantize_per_channel_affine."""
    return record_op("Fake_quantize_per_channel_affine", [x] + list(args), kwargs)


@register_op("Fake_quantize_per_tensor_affine", "ai.onnx")
def fake_quantize_per_tensor_affine(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fake_quantize_per_tensor_affine."""
    return record_op("Fake_quantize_per_tensor_affine", [x] + list(args), kwargs)


@register_op("Fbgemm_linear_fp16_weight", "ai.onnx")
def fbgemm_linear_fp16_weight(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fbgemm_linear_fp16_weight."""
    return record_op("Fbgemm_linear_fp16_weight", [x] + list(args), kwargs)


@register_op("Fbgemm_linear_fp16_weight_fp32_activation", "ai.onnx")
def fbgemm_linear_fp16_weight_fp32_activation(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fbgemm_linear_fp16_weight_fp32_activation."""
    return record_op("Fbgemm_linear_fp16_weight_fp32_activation", [x] + list(args), kwargs)


@register_op("Fbgemm_linear_int8_weight", "ai.onnx")
def fbgemm_linear_int8_weight(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fbgemm_linear_int8_weight."""
    return record_op("Fbgemm_linear_int8_weight", [x] + list(args), kwargs)


@register_op("Fbgemm_linear_int8_weight_fp32_activation", "ai.onnx")
def fbgemm_linear_int8_weight_fp32_activation(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fbgemm_linear_int8_weight_fp32_activation."""
    return record_op("Fbgemm_linear_int8_weight_fp32_activation", [x] + list(args), kwargs)


@register_op("Fbgemm_linear_quantize_weight", "ai.onnx")
def fbgemm_linear_quantize_weight(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fbgemm_linear_quantize_weight."""
    return record_op("Fbgemm_linear_quantize_weight", [x] + list(args), kwargs)


@register_op("Fbgemm_pack_gemm_matrix_fp16", "ai.onnx")
def fbgemm_pack_gemm_matrix_fp16(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fbgemm_pack_gemm_matrix_fp16."""
    return record_op("Fbgemm_pack_gemm_matrix_fp16", [x] + list(args), kwargs)


@register_op("Fbgemm_pack_quantized_matrix", "ai.onnx")
def fbgemm_pack_quantized_matrix(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fbgemm_pack_quantized_matrix."""
    return record_op("Fbgemm_pack_quantized_matrix", [x] + list(args), kwargs)


@register_op("Feature_alpha_dropout", "ai.onnx")
def feature_alpha_dropout(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Feature_alpha_dropout."""
    return record_op("Feature_alpha_dropout", [x] + list(args), kwargs)


@register_op("Feature_alpha_dropout_", "ai.onnx")
def feature_alpha_dropout_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Feature_alpha_dropout_."""
    return record_op("Feature_alpha_dropout_", [x] + list(args), kwargs)


@register_op("Feature_dropout", "ai.onnx")
def feature_dropout(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Feature_dropout."""
    return record_op("Feature_dropout", [x] + list(args), kwargs)


@register_op("Feature_dropout_", "ai.onnx")
def feature_dropout_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Feature_dropout_."""
    return record_op("Feature_dropout_", [x] + list(args), kwargs)


@register_op("Fill", "ai.onnx")
def fill(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fill."""
    return record_op("Fill", [x] + list(args), kwargs)


@register_op("Fill_", "ai.onnx")
def fill_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fill_."""
    return record_op("Fill_", [x] + list(args), kwargs)


@register_op("Fix", "ai.onnx")
def fix(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fix."""
    return record_op("Fix", [x] + list(args), kwargs)


@register_op("Fix_", "ai.onnx")
def fix_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fix_."""
    return record_op("Fix_", [x] + list(args), kwargs)


@register_op("Flatten", "ai.onnx")
def flatten(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Flatten."""
    return record_op("Flatten", [x] + list(args), kwargs)


@register_op("Flip", "ai.onnx")
def flip(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Flip."""
    return record_op("Flip", [x] + list(args), kwargs)


@register_op("Fliplr", "ai.onnx")
def fliplr(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fliplr."""
    return record_op("Fliplr", [x] + list(args), kwargs)


@register_op("Flipud", "ai.onnx")
def flipud(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Flipud."""
    return record_op("Flipud", [x] + list(args), kwargs)


@register_op("Float_power", "ai.onnx")
def float_power(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Float_power."""
    return record_op("Float_power", [x] + list(args), kwargs)


@register_op("Floor_", "ai.onnx")
def floor_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Floor_."""
    return record_op("Floor_", [x] + list(args), kwargs)


@register_op("Floor_divide", "ai.onnx")
def floor_divide(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Floor_divide."""
    return record_op("Floor_divide", [x] + list(args), kwargs)


@register_op("Fmax", "ai.onnx")
def fmax(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fmax."""
    return record_op("Fmax", [x] + list(args), kwargs)


@register_op("Fmin", "ai.onnx")
def fmin(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fmin."""
    return record_op("Fmin", [x] + list(args), kwargs)


@register_op("Fmod", "ai.onnx")
def fmod(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fmod."""
    return record_op("Fmod", [x] + list(args), kwargs)


@register_op("Frac", "ai.onnx")
def frac(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Frac."""
    return record_op("Frac", [x] + list(args), kwargs)


@register_op("Frac_", "ai.onnx")
def frac_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Frac_."""
    return record_op("Frac_", [x] + list(args), kwargs)


@register_op("Frexp", "ai.onnx")
def frexp(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Frexp."""
    return record_op("Frexp", [x] + list(args), kwargs)


@register_op("Frobenius_norm", "ai.onnx")
def frobenius_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Frobenius_norm."""
    return record_op("Frobenius_norm", [x] + list(args), kwargs)


@register_op("From_file", "ai.onnx")
def from_file(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute From_file."""
    return record_op("From_file", [x] + list(args), kwargs)


@register_op("From_numpy", "ai.onnx")
def from_numpy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute From_numpy."""
    return record_op("From_numpy", [x] + list(args), kwargs)


@register_op("Frombuffer", "ai.onnx")
def frombuffer(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Frombuffer."""
    return record_op("Frombuffer", [x] + list(args), kwargs)


@register_op("Full", "ai.onnx")
def full(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Full."""
    return record_op("Full", [x] + list(args), kwargs)


@register_op("Full_like", "ai.onnx")
def full_like(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Full_like."""
    return record_op("Full_like", [x] + list(args), kwargs)


@register_op("Fused_moving_avg_obs_fake_quant", "ai.onnx")
def fused_moving_avg_obs_fake_quant(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fused_moving_avg_obs_fake_quant."""
    return record_op("Fused_moving_avg_obs_fake_quant", [x] + list(args), kwargs)


@register_op("Gcd", "ai.onnx")
def gcd(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Gcd."""
    return record_op("Gcd", [x] + list(args), kwargs)


@register_op("Gcd_", "ai.onnx")
def gcd_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Gcd_."""
    return record_op("Gcd_", [x] + list(args), kwargs)


@register_op("Ge", "ai.onnx")
def ge(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Ge."""
    return record_op("Ge", [x] + list(args), kwargs)


@register_op("Geqrf", "ai.onnx")
def geqrf(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Geqrf."""
    return record_op("Geqrf", [x] + list(args), kwargs)


@register_op("Ger", "ai.onnx")
def ger(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Ger."""
    return record_op("Ger", [x] + list(args), kwargs)


@register_op("Get_device", "ai.onnx")
def get_device(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_device."""
    return record_op("Get_device", [x] + list(args), kwargs)


@register_op("Gradient", "ai.onnx")
def gradient(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Gradient."""
    return record_op("Gradient", [x] + list(args), kwargs)


@register_op("Greater_equal", "ai.onnx")
def greater_equal(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Greater_equal."""
    return record_op("Greater_equal", [x] + list(args), kwargs)


@register_op("Grid_sampler", "ai.onnx")
def grid_sampler(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Grid_sampler."""
    return record_op("Grid_sampler", [x] + list(args), kwargs)


@register_op("Grid_sampler_2d", "ai.onnx")
def grid_sampler_2d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Grid_sampler_2d."""
    return record_op("Grid_sampler_2d", [x] + list(args), kwargs)


@register_op("Grid_sampler_3d", "ai.onnx")
def grid_sampler_3d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Grid_sampler_3d."""
    return record_op("Grid_sampler_3d", [x] + list(args), kwargs)


@register_op("Group_norm", "ai.onnx")
def group_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Group_norm."""
    return record_op("Group_norm", [x] + list(args), kwargs)


@register_op("Gru_cell", "ai.onnx")
def gru_cell(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Gru_cell."""
    return record_op("Gru_cell", [x] + list(args), kwargs)


@register_op("Gt", "ai.onnx")
def gt(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Gt."""
    return record_op("Gt", [x] + list(args), kwargs)


@register_op("Hamming_window", "ai.onnx")
def hamming_window(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Hamming_window."""
    return record_op("Hamming_window", [x] + list(args), kwargs)


@register_op("Hann_window", "ai.onnx")
def hann_window(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Hann_window."""
    return record_op("Hann_window", [x] + list(args), kwargs)


@register_op("Hardshrink", "ai.onnx")
def hardshrink(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Hardshrink."""
    return record_op("Hardshrink", [x] + list(args), kwargs)


@register_op("Hash_tensor", "ai.onnx")
def hash_tensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Hash_tensor."""
    return record_op("Hash_tensor", [x] + list(args), kwargs)


@register_op("Heaviside", "ai.onnx")
def heaviside(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Heaviside."""
    return record_op("Heaviside", [x] + list(args), kwargs)


@register_op("Hinge_embedding_loss", "ai.onnx")
def hinge_embedding_loss(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Hinge_embedding_loss."""
    return record_op("Hinge_embedding_loss", [x] + list(args), kwargs)


@register_op("Histc", "ai.onnx")
def histc(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Histc."""
    return record_op("Histc", [x] + list(args), kwargs)


@register_op("Histogram", "ai.onnx")
def histogram(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Histogram."""
    return record_op("Histogram", [x] + list(args), kwargs)


@register_op("Histogramdd", "ai.onnx")
def histogramdd(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Histogramdd."""
    return record_op("Histogramdd", [x] + list(args), kwargs)


@register_op("Hsmm", "ai.onnx")
def hsmm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Hsmm."""
    return record_op("Hsmm", [x] + list(args), kwargs)


@register_op("Hsplit", "ai.onnx")
def hsplit(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Hsplit."""
    return record_op("Hsplit", [x] + list(args), kwargs)


@register_op("Hspmm", "ai.onnx")
def hspmm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Hspmm."""
    return record_op("Hspmm", [x] + list(args), kwargs)


@register_op("Hstack", "ai.onnx")
def hstack(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Hstack."""
    return record_op("Hstack", [x] + list(args), kwargs)


@register_op("Hypot", "ai.onnx")
def hypot(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Hypot."""
    return record_op("Hypot", [x] + list(args), kwargs)


@register_op("I0", "ai.onnx")
def i0(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute I0."""
    return record_op("I0", [x] + list(args), kwargs)


@register_op("I0_", "ai.onnx")
def i0_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute I0_."""
    return record_op("I0_", [x] + list(args), kwargs)


@register_op("Igamma", "ai.onnx")
def igamma(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Igamma."""
    return record_op("Igamma", [x] + list(args), kwargs)


@register_op("Igammac", "ai.onnx")
def igammac(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Igammac."""
    return record_op("Igammac", [x] + list(args), kwargs)


@register_op("Imag", "ai.onnx")
def imag(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Imag."""
    return record_op("Imag", [x] + list(args), kwargs)


@register_op("Index_add", "ai.onnx")
def index_add(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Index_add."""
    return record_op("Index_add", [x] + list(args), kwargs)


@register_op("Index_copy", "ai.onnx")
def index_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Index_copy."""
    return record_op("Index_copy", [x] + list(args), kwargs)


@register_op("Index_fill", "ai.onnx")
def index_fill(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Index_fill."""
    return record_op("Index_fill", [x] + list(args), kwargs)


@register_op("Index_put", "ai.onnx")
def index_put(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Index_put."""
    return record_op("Index_put", [x] + list(args), kwargs)


@register_op("Index_put_", "ai.onnx")
def index_put_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Index_put_."""
    return record_op("Index_put_", [x] + list(args), kwargs)


@register_op("Index_reduce", "ai.onnx")
def index_reduce(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Index_reduce."""
    return record_op("Index_reduce", [x] + list(args), kwargs)


@register_op("Index_select", "ai.onnx")
def index_select(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Index_select."""
    return record_op("Index_select", [x] + list(args), kwargs)


@register_op("Indices_copy", "ai.onnx")
def indices_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Indices_copy."""
    return record_op("Indices_copy", [x] + list(args), kwargs)


@register_op("Inner", "ai.onnx")
def inner(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Inner."""
    return record_op("Inner", [x] + list(args), kwargs)


@register_op("Instance_norm", "ai.onnx")
def instance_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Instance_norm."""
    return record_op("Instance_norm", [x] + list(args), kwargs)


@register_op("Int_repr", "ai.onnx")
def int_repr(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Int_repr."""
    return record_op("Int_repr", [x] + list(args), kwargs)


@register_op("Inverse", "ai.onnx")
def inverse(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Inverse."""
    return record_op("Inverse", [x] + list(args), kwargs)


@register_op("Is_complex", "ai.onnx")
def is_complex(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_complex."""
    return record_op("Is_complex", [x] + list(args), kwargs)


@register_op("Is_conj", "ai.onnx")
def is_conj(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_conj."""
    return record_op("Is_conj", [x] + list(args), kwargs)


@register_op("Is_distributed", "ai.onnx")
def is_distributed(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_distributed."""
    return record_op("Is_distributed", [x] + list(args), kwargs)


@register_op("Is_floating_point", "ai.onnx")
def is_floating_point(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_floating_point."""
    return record_op("Is_floating_point", [x] + list(args), kwargs)


@register_op("Is_inference", "ai.onnx")
def is_inference(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_inference."""
    return record_op("Is_inference", [x] + list(args), kwargs)


@register_op("Is_neg", "ai.onnx")
def is_neg(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_neg."""
    return record_op("Is_neg", [x] + list(args), kwargs)


@register_op("Is_nonzero", "ai.onnx")
def is_nonzero(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_nonzero."""
    return record_op("Is_nonzero", [x] + list(args), kwargs)


@register_op("Is_same_size", "ai.onnx")
def is_same_size(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_same_size."""
    return record_op("Is_same_size", [x] + list(args), kwargs)


@register_op("Is_signed", "ai.onnx")
def is_signed(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_signed."""
    return record_op("Is_signed", [x] + list(args), kwargs)


@register_op("Is_vulkan_available", "ai.onnx")
def is_vulkan_available(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_vulkan_available."""
    return record_op("Is_vulkan_available", [x] + list(args), kwargs)


@register_op("Isclose", "ai.onnx")
def isclose(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Isclose."""
    return record_op("Isclose", [x] + list(args), kwargs)


@register_op("Isfinite", "ai.onnx")
def isfinite(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Isfinite."""
    return record_op("Isfinite", [x] + list(args), kwargs)


@register_op("Isin", "ai.onnx")
def isin(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Isin."""
    return record_op("Isin", [x] + list(args), kwargs)


@register_op("Isneginf", "ai.onnx")
def isneginf(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Isneginf."""
    return record_op("Isneginf", [x] + list(args), kwargs)


@register_op("Isposinf", "ai.onnx")
def isposinf(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Isposinf."""
    return record_op("Isposinf", [x] + list(args), kwargs)


@register_op("Isreal", "ai.onnx")
def isreal(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Isreal."""
    return record_op("Isreal", [x] + list(args), kwargs)


@register_op("Istft", "ai.onnx")
def istft(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Istft."""
    return record_op("Istft", [x] + list(args), kwargs)


@register_op("Kaiser_window", "ai.onnx")
def kaiser_window(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Kaiser_window."""
    return record_op("Kaiser_window", [x] + list(args), kwargs)


@register_op("Kl_div", "ai.onnx")
def kl_div(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Kl_div."""
    return record_op("Kl_div", [x] + list(args), kwargs)


@register_op("Kron", "ai.onnx")
def kron(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Kron."""
    return record_op("Kron", [x] + list(args), kwargs)


@register_op("Kthvalue", "ai.onnx")
def kthvalue(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Kthvalue."""
    return record_op("Kthvalue", [x] + list(args), kwargs)


@register_op("Layer_norm", "ai.onnx")
def layer_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Layer_norm."""
    return record_op("Layer_norm", [x] + list(args), kwargs)


@register_op("Lcm", "ai.onnx")
def lcm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Lcm."""
    return record_op("Lcm", [x] + list(args), kwargs)


@register_op("Lcm_", "ai.onnx")
def lcm_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Lcm_."""
    return record_op("Lcm_", [x] + list(args), kwargs)


@register_op("Ldexp", "ai.onnx")
def ldexp(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Ldexp."""
    return record_op("Ldexp", [x] + list(args), kwargs)


@register_op("Ldexp_", "ai.onnx")
def ldexp_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Ldexp_."""
    return record_op("Ldexp_", [x] + list(args), kwargs)


@register_op("Le", "ai.onnx")
def le(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Le."""
    return record_op("Le", [x] + list(args), kwargs)


@register_op("Lerp", "ai.onnx")
def lerp(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Lerp."""
    return record_op("Lerp", [x] + list(args), kwargs)


@register_op("Less_equal", "ai.onnx")
def less_equal(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Less_equal."""
    return record_op("Less_equal", [x] + list(args), kwargs)


@register_op("Lgamma", "ai.onnx")
def lgamma(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Lgamma."""
    return record_op("Lgamma", [x] + list(args), kwargs)


@register_op("Linspace", "ai.onnx")
def linspace(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Linspace."""
    return record_op("Linspace", [x] + list(args), kwargs)


@register_op("Log", "ai.onnx")
def log(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Log."""
    return record_op("Log", [x] + list(args), kwargs)


@register_op("Log10", "ai.onnx")
def log10(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Log10."""
    return record_op("Log10", [x] + list(args), kwargs)


@register_op("Log10_", "ai.onnx")
def log10_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Log10_."""
    return record_op("Log10_", [x] + list(args), kwargs)


@register_op("Log1p", "ai.onnx")
def log1p(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Log1p."""
    return record_op("Log1p", [x] + list(args), kwargs)


@register_op("Log1p_", "ai.onnx")
def log1p_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Log1p_."""
    return record_op("Log1p_", [x] + list(args), kwargs)


@register_op("Log2", "ai.onnx")
def log2(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Log2."""
    return record_op("Log2", [x] + list(args), kwargs)


@register_op("Log2_", "ai.onnx")
def log2_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Log2_."""
    return record_op("Log2_", [x] + list(args), kwargs)


@register_op("Log_", "ai.onnx")
def log_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Log_."""
    return record_op("Log_", [x] + list(args), kwargs)


@register_op("Logaddexp", "ai.onnx")
def logaddexp(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Logaddexp."""
    return record_op("Logaddexp", [x] + list(args), kwargs)


@register_op("Logaddexp2", "ai.onnx")
def logaddexp2(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Logaddexp2."""
    return record_op("Logaddexp2", [x] + list(args), kwargs)


@register_op("Logcumsumexp", "ai.onnx")
def logcumsumexp(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Logcumsumexp."""
    return record_op("Logcumsumexp", [x] + list(args), kwargs)


@register_op("Logdet", "ai.onnx")
def logdet(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Logdet."""
    return record_op("Logdet", [x] + list(args), kwargs)


@register_op("Logical_and", "ai.onnx")
def logical_and(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Logical_and."""
    return record_op("Logical_and", [x] + list(args), kwargs)


@register_op("Logical_not", "ai.onnx")
def logical_not(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Logical_not."""
    return record_op("Logical_not", [x] + list(args), kwargs)


@register_op("Logical_or", "ai.onnx")
def logical_or(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Logical_or."""
    return record_op("Logical_or", [x] + list(args), kwargs)


@register_op("Logical_xor", "ai.onnx")
def logical_xor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Logical_xor."""
    return record_op("Logical_xor", [x] + list(args), kwargs)


@register_op("Logit", "ai.onnx")
def logit(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Logit."""
    return record_op("Logit", [x] + list(args), kwargs)


@register_op("Logit_", "ai.onnx")
def logit_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Logit_."""
    return record_op("Logit_", [x] + list(args), kwargs)


@register_op("Logspace", "ai.onnx")
def logspace(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Logspace."""
    return record_op("Logspace", [x] + list(args), kwargs)


@register_op("Logsumexp", "ai.onnx")
def logsumexp(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Logsumexp."""
    return record_op("Logsumexp", [x] + list(args), kwargs)


@register_op("Lstm_cell", "ai.onnx")
def lstm_cell(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Lstm_cell."""
    return record_op("Lstm_cell", [x] + list(args), kwargs)


@register_op("Lt", "ai.onnx")
def lt(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Lt."""
    return record_op("Lt", [x] + list(args), kwargs)


@register_op("Lu_solve", "ai.onnx")
def lu_solve(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Lu_solve."""
    return record_op("Lu_solve", [x] + list(args), kwargs)


@register_op("Lu_unpack", "ai.onnx")
def lu_unpack(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Lu_unpack."""
    return record_op("Lu_unpack", [x] + list(args), kwargs)


@register_op("Margin_ranking_loss", "ai.onnx")
def margin_ranking_loss(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Margin_ranking_loss."""
    return record_op("Margin_ranking_loss", [x] + list(args), kwargs)


@register_op("Masked_fill", "ai.onnx")
def masked_fill(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Masked_fill."""
    return record_op("Masked_fill", [x] + list(args), kwargs)


@register_op("Masked_scatter", "ai.onnx")
def masked_scatter(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Masked_scatter."""
    return record_op("Masked_scatter", [x] + list(args), kwargs)


@register_op("Masked_select", "ai.onnx")
def masked_select(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Masked_select."""
    return record_op("Masked_select", [x] + list(args), kwargs)


@register_op("Matrix_exp", "ai.onnx")
def matrix_exp(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Matrix_exp."""
    return record_op("Matrix_exp", [x] + list(args), kwargs)


@register_op("Matrix_power", "ai.onnx")
def matrix_power(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Matrix_power."""
    return record_op("Matrix_power", [x] + list(args), kwargs)


@register_op("Max_pool1d", "ai.onnx")
def max_pool1d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Max_pool1d."""
    return record_op("Max_pool1d", [x] + list(args), kwargs)


@register_op("Max_pool1d_with_indices", "ai.onnx")
def max_pool1d_with_indices(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Max_pool1d_with_indices."""
    return record_op("Max_pool1d_with_indices", [x] + list(args), kwargs)


@register_op("Max_pool2d", "ai.onnx")
def max_pool2d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Max_pool2d."""
    return record_op("Max_pool2d", [x] + list(args), kwargs)


@register_op("Max_pool3d", "ai.onnx")
def max_pool3d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Max_pool3d."""
    return record_op("Max_pool3d", [x] + list(args), kwargs)


@register_op("Maximum", "ai.onnx")
def maximum(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Maximum."""
    return record_op("Maximum", [x] + list(args), kwargs)


@register_op("Median", "ai.onnx")
def median(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Median."""
    return record_op("Median", [x] + list(args), kwargs)


@register_op("Meshgrid", "ai.onnx")
def meshgrid(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Meshgrid."""
    return record_op("Meshgrid", [x] + list(args), kwargs)


@register_op("Minimum", "ai.onnx")
def minimum(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Minimum."""
    return record_op("Minimum", [x] + list(args), kwargs)


@register_op("Miopen_batch_norm", "ai.onnx")
def miopen_batch_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Miopen_batch_norm."""
    return record_op("Miopen_batch_norm", [x] + list(args), kwargs)


@register_op("Miopen_convolution", "ai.onnx")
def miopen_convolution(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Miopen_convolution."""
    return record_op("Miopen_convolution", [x] + list(args), kwargs)


@register_op("Miopen_convolution_add_relu", "ai.onnx")
def miopen_convolution_add_relu(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Miopen_convolution_add_relu."""
    return record_op("Miopen_convolution_add_relu", [x] + list(args), kwargs)


@register_op("Miopen_convolution_relu", "ai.onnx")
def miopen_convolution_relu(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Miopen_convolution_relu."""
    return record_op("Miopen_convolution_relu", [x] + list(args), kwargs)


@register_op("Miopen_convolution_transpose", "ai.onnx")
def miopen_convolution_transpose(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Miopen_convolution_transpose."""
    return record_op("Miopen_convolution_transpose", [x] + list(args), kwargs)


@register_op("Miopen_ctc_loss", "ai.onnx")
def miopen_ctc_loss(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Miopen_ctc_loss."""
    return record_op("Miopen_ctc_loss", [x] + list(args), kwargs)


@register_op("Miopen_depthwise_convolution", "ai.onnx")
def miopen_depthwise_convolution(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Miopen_depthwise_convolution."""
    return record_op("Miopen_depthwise_convolution", [x] + list(args), kwargs)


@register_op("Miopen_rnn", "ai.onnx")
def miopen_rnn(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Miopen_rnn."""
    return record_op("Miopen_rnn", [x] + list(args), kwargs)


@register_op("Mkldnn_adaptive_avg_pool2d", "ai.onnx")
def mkldnn_adaptive_avg_pool2d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Mkldnn_adaptive_avg_pool2d."""
    return record_op("Mkldnn_adaptive_avg_pool2d", [x] + list(args), kwargs)


@register_op("Mkldnn_convolution", "ai.onnx")
def mkldnn_convolution(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Mkldnn_convolution."""
    return record_op("Mkldnn_convolution", [x] + list(args), kwargs)


@register_op("Mkldnn_linear_backward_weights", "ai.onnx")
def mkldnn_linear_backward_weights(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Mkldnn_linear_backward_weights."""
    return record_op("Mkldnn_linear_backward_weights", [x] + list(args), kwargs)


@register_op("Mkldnn_max_pool2d", "ai.onnx")
def mkldnn_max_pool2d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Mkldnn_max_pool2d."""
    return record_op("Mkldnn_max_pool2d", [x] + list(args), kwargs)


@register_op("Mkldnn_max_pool3d", "ai.onnx")
def mkldnn_max_pool3d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Mkldnn_max_pool3d."""
    return record_op("Mkldnn_max_pool3d", [x] + list(args), kwargs)


@register_op("Mkldnn_rnn_layer", "ai.onnx")
def mkldnn_rnn_layer(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Mkldnn_rnn_layer."""
    return record_op("Mkldnn_rnn_layer", [x] + list(args), kwargs)


@register_op("Mm", "ai.onnx")
def mm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Mm."""
    return record_op("Mm", [x] + list(args), kwargs)


@register_op("Mode", "ai.onnx")
def mode(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Mode."""
    return record_op("Mode", [x] + list(args), kwargs)


@register_op("Moveaxis", "ai.onnx")
def moveaxis(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Moveaxis."""
    return record_op("Moveaxis", [x] + list(args), kwargs)


@register_op("Movedim", "ai.onnx")
def movedim(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Movedim."""
    return record_op("Movedim", [x] + list(args), kwargs)


@register_op("Msort", "ai.onnx")
def msort(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Msort."""
    return record_op("Msort", [x] + list(args), kwargs)


@register_op("Multiply", "ai.onnx")
def multiply(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Multiply."""
    return record_op("Multiply", [x] + list(args), kwargs)


@register_op("Mv", "ai.onnx")
def mv(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Mv."""
    return record_op("Mv", [x] + list(args), kwargs)


@register_op("Mvlgamma", "ai.onnx")
def mvlgamma(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Mvlgamma."""
    return record_op("Mvlgamma", [x] + list(args), kwargs)


@register_op("Nan_to_num", "ai.onnx")
def nan_to_num(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Nan_to_num."""
    return record_op("Nan_to_num", [x] + list(args), kwargs)


@register_op("Nan_to_num_", "ai.onnx")
def nan_to_num_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Nan_to_num_."""
    return record_op("Nan_to_num_", [x] + list(args), kwargs)


@register_op("Nanmean", "ai.onnx")
def nanmean(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Nanmean."""
    return record_op("Nanmean", [x] + list(args), kwargs)


@register_op("Nanmedian", "ai.onnx")
def nanmedian(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Nanmedian."""
    return record_op("Nanmedian", [x] + list(args), kwargs)


@register_op("Nanquantile", "ai.onnx")
def nanquantile(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Nanquantile."""
    return record_op("Nanquantile", [x] + list(args), kwargs)


@register_op("Nansum", "ai.onnx")
def nansum(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Nansum."""
    return record_op("Nansum", [x] + list(args), kwargs)


@register_op("Narrow", "ai.onnx")
def narrow(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Narrow."""
    return record_op("Narrow", [x] + list(args), kwargs)


@register_op("Narrow_copy", "ai.onnx")
def narrow_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Narrow_copy."""
    return record_op("Narrow_copy", [x] + list(args), kwargs)


@register_op("Native_batch_norm", "ai.onnx")
def native_batch_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Native_batch_norm."""
    return record_op("Native_batch_norm", [x] + list(args), kwargs)


@register_op("Native_channel_shuffle", "ai.onnx")
def native_channel_shuffle(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Native_channel_shuffle."""
    return record_op("Native_channel_shuffle", [x] + list(args), kwargs)


@register_op("Native_dropout", "ai.onnx")
def native_dropout(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Native_dropout."""
    return record_op("Native_dropout", [x] + list(args), kwargs)


@register_op("Native_group_norm", "ai.onnx")
def native_group_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Native_group_norm."""
    return record_op("Native_group_norm", [x] + list(args), kwargs)


@register_op("Native_layer_norm", "ai.onnx")
def native_layer_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Native_layer_norm."""
    return record_op("Native_layer_norm", [x] + list(args), kwargs)


@register_op("Native_norm", "ai.onnx")
def native_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Native_norm."""
    return record_op("Native_norm", [x] + list(args), kwargs)


@register_op("Ne", "ai.onnx")
def ne(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Ne."""
    return record_op("Ne", [x] + list(args), kwargs)


@register_op("Neg_", "ai.onnx")
def neg_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Neg_."""
    return record_op("Neg_", [x] + list(args), kwargs)


@register_op("Negative", "ai.onnx")
def negative(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Negative."""
    return record_op("Negative", [x] + list(args), kwargs)


@register_op("Negative_", "ai.onnx")
def negative_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Negative_."""
    return record_op("Negative_", [x] + list(args), kwargs)


@register_op("Nextafter", "ai.onnx")
def nextafter(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Nextafter."""
    return record_op("Nextafter", [x] + list(args), kwargs)


@register_op("Nonzero", "ai.onnx")
def nonzero(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Nonzero."""
    return record_op("Nonzero", [x] + list(args), kwargs)


@register_op("Nonzero_static", "ai.onnx")
def nonzero_static(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Nonzero_static."""
    return record_op("Nonzero_static", [x] + list(args), kwargs)


@register_op("Norm", "ai.onnx")
def norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Norm."""
    return record_op("Norm", [x] + list(args), kwargs)


@register_op("Norm_except_dim", "ai.onnx")
def norm_except_dim(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Norm_except_dim."""
    return record_op("Norm_except_dim", [x] + list(args), kwargs)


@register_op("Normal", "ai.onnx")
def normal(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Normal."""
    return record_op("Normal", [x] + list(args), kwargs)


@register_op("Not_equal", "ai.onnx")
def not_equal(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Not_equal."""
    return record_op("Not_equal", [x] + list(args), kwargs)


@register_op("Nuclear_norm", "ai.onnx")
def nuclear_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Nuclear_norm."""
    return record_op("Nuclear_norm", [x] + list(args), kwargs)


@register_op("Numel", "ai.onnx")
def numel(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Numel."""
    return record_op("Numel", [x] + list(args), kwargs)


@register_op("Ones_like", "ai.onnx")
def ones_like(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Ones_like."""
    return record_op("Ones_like", [x] + list(args), kwargs)


@register_op("Orgqr", "ai.onnx")
def orgqr(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Orgqr."""
    return record_op("Orgqr", [x] + list(args), kwargs)


@register_op("Ormqr", "ai.onnx")
def ormqr(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Ormqr."""
    return record_op("Ormqr", [x] + list(args), kwargs)


@register_op("Outer", "ai.onnx")
def outer(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Outer."""
    return record_op("Outer", [x] + list(args), kwargs)


@register_op("Pairwise_distance", "ai.onnx")
def pairwise_distance(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Pairwise_distance."""
    return record_op("Pairwise_distance", [x] + list(args), kwargs)


@register_op("Pdist", "ai.onnx")
def pdist(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Pdist."""
    return record_op("Pdist", [x] + list(args), kwargs)


@register_op("Permute", "ai.onnx")
def permute(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Permute."""
    return record_op("Permute", [x] + list(args), kwargs)


@register_op("Permute_copy", "ai.onnx")
def permute_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Permute_copy."""
    return record_op("Permute_copy", [x] + list(args), kwargs)


@register_op("Pinverse", "ai.onnx")
def pinverse(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Pinverse."""
    return record_op("Pinverse", [x] + list(args), kwargs)


@register_op("Pixel_shuffle", "ai.onnx")
def pixel_shuffle(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Pixel_shuffle."""
    return record_op("Pixel_shuffle", [x] + list(args), kwargs)


@register_op("Pixel_unshuffle", "ai.onnx")
def pixel_unshuffle(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Pixel_unshuffle."""
    return record_op("Pixel_unshuffle", [x] + list(args), kwargs)


@register_op("Poisson", "ai.onnx")
def poisson(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Poisson."""
    return record_op("Poisson", [x] + list(args), kwargs)


@register_op("Poisson_nll_loss", "ai.onnx")
def poisson_nll_loss(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Poisson_nll_loss."""
    return record_op("Poisson_nll_loss", [x] + list(args), kwargs)


@register_op("Polar", "ai.onnx")
def polar(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Polar."""
    return record_op("Polar", [x] + list(args), kwargs)


@register_op("Polygamma", "ai.onnx")
def polygamma(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Polygamma."""
    return record_op("Polygamma", [x] + list(args), kwargs)


@register_op("Positive", "ai.onnx")
def positive(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Positive."""
    return record_op("Positive", [x] + list(args), kwargs)


@register_op("Prod", "ai.onnx")
def prod(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Prod."""
    return record_op("Prod", [x] + list(args), kwargs)


@register_op("Promote_types", "ai.onnx")
def promote_types(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Promote_types."""
    return record_op("Promote_types", [x] + list(args), kwargs)


@register_op("Put", "ai.onnx")
def put(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Put."""
    return record_op("Put", [x] + list(args), kwargs)


@register_op("Q_per_channel_axis", "ai.onnx")
def q_per_channel_axis(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Q_per_channel_axis."""
    return record_op("Q_per_channel_axis", [x] + list(args), kwargs)


@register_op("Q_per_channel_scales", "ai.onnx")
def q_per_channel_scales(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Q_per_channel_scales."""
    return record_op("Q_per_channel_scales", [x] + list(args), kwargs)


@register_op("Q_per_channel_zero_points", "ai.onnx")
def q_per_channel_zero_points(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Q_per_channel_zero_points."""
    return record_op("Q_per_channel_zero_points", [x] + list(args), kwargs)


@register_op("Q_scale", "ai.onnx")
def q_scale(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Q_scale."""
    return record_op("Q_scale", [x] + list(args), kwargs)


@register_op("Q_zero_point", "ai.onnx")
def q_zero_point(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Q_zero_point."""
    return record_op("Q_zero_point", [x] + list(args), kwargs)


@register_op("Qr", "ai.onnx")
def qr(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Qr."""
    return record_op("Qr", [x] + list(args), kwargs)


@register_op("Quantile", "ai.onnx")
def quantile(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quantile."""
    return record_op("Quantile", [x] + list(args), kwargs)


@register_op("Quantize_per_channel", "ai.onnx")
def quantize_per_channel(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quantize_per_channel."""
    return record_op("Quantize_per_channel", [x] + list(args), kwargs)


@register_op("Quantize_per_tensor", "ai.onnx")
def quantize_per_tensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quantize_per_tensor."""
    return record_op("Quantize_per_tensor", [x] + list(args), kwargs)


@register_op("Quantize_per_tensor_dynamic", "ai.onnx")
def quantize_per_tensor_dynamic(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quantize_per_tensor_dynamic."""
    return record_op("Quantize_per_tensor_dynamic", [x] + list(args), kwargs)


@register_op("Quantized_batch_norm", "ai.onnx")
def quantized_batch_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quantized_batch_norm."""
    return record_op("Quantized_batch_norm", [x] + list(args), kwargs)


@register_op("Quantized_gru_cell", "ai.onnx")
def quantized_gru_cell(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quantized_gru_cell."""
    return record_op("Quantized_gru_cell", [x] + list(args), kwargs)


@register_op("Quantized_lstm_cell", "ai.onnx")
def quantized_lstm_cell(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quantized_lstm_cell."""
    return record_op("Quantized_lstm_cell", [x] + list(args), kwargs)


@register_op("Quantized_max_pool1d", "ai.onnx")
def quantized_max_pool1d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quantized_max_pool1d."""
    return record_op("Quantized_max_pool1d", [x] + list(args), kwargs)


@register_op("Quantized_max_pool2d", "ai.onnx")
def quantized_max_pool2d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quantized_max_pool2d."""
    return record_op("Quantized_max_pool2d", [x] + list(args), kwargs)


@register_op("Quantized_max_pool3d", "ai.onnx")
def quantized_max_pool3d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quantized_max_pool3d."""
    return record_op("Quantized_max_pool3d", [x] + list(args), kwargs)


@register_op("Quantized_rnn_relu_cell", "ai.onnx")
def quantized_rnn_relu_cell(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quantized_rnn_relu_cell."""
    return record_op("Quantized_rnn_relu_cell", [x] + list(args), kwargs)


@register_op("Quantized_rnn_tanh_cell", "ai.onnx")
def quantized_rnn_tanh_cell(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quantized_rnn_tanh_cell."""
    return record_op("Quantized_rnn_tanh_cell", [x] + list(args), kwargs)


@register_op("Rad2deg", "ai.onnx")
def rad2deg(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rad2deg."""
    return record_op("Rad2deg", [x] + list(args), kwargs)


@register_op("Rad2deg_", "ai.onnx")
def rad2deg_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rad2deg_."""
    return record_op("Rad2deg_", [x] + list(args), kwargs)


@register_op("Rand_like", "ai.onnx")
def rand_like(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rand_like."""
    return record_op("Rand_like", [x] + list(args), kwargs)


@register_op("Randint", "ai.onnx")
def randint(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Randint."""
    return record_op("Randint", [x] + list(args), kwargs)


@register_op("Randint_like", "ai.onnx")
def randint_like(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Randint_like."""
    return record_op("Randint_like", [x] + list(args), kwargs)


@register_op("Randn_like", "ai.onnx")
def randn_like(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Randn_like."""
    return record_op("Randn_like", [x] + list(args), kwargs)


@register_op("Randperm", "ai.onnx")
def randperm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Randperm."""
    return record_op("Randperm", [x] + list(args), kwargs)


@register_op("Range", "ai.onnx")
def range(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Range."""
    return record_op("Range", [x] + list(args), kwargs)


@register_op("Ravel", "ai.onnx")
def ravel(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Ravel."""
    return record_op("Ravel", [x] + list(args), kwargs)


@register_op("Real", "ai.onnx")
def real(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Real."""
    return record_op("Real", [x] + list(args), kwargs)


@register_op("Reciprocal_", "ai.onnx")
def reciprocal_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Reciprocal_."""
    return record_op("Reciprocal_", [x] + list(args), kwargs)


@register_op("Relu_", "ai.onnx")
def relu_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Relu_."""
    return record_op("Relu_", [x] + list(args), kwargs)


@register_op("Remainder", "ai.onnx")
def remainder(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Remainder."""
    return record_op("Remainder", [x] + list(args), kwargs)


@register_op("Renorm", "ai.onnx")
def renorm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Renorm."""
    return record_op("Renorm", [x] + list(args), kwargs)


@register_op("Repeat_interleave", "ai.onnx")
def repeat_interleave(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Repeat_interleave."""
    return record_op("Repeat_interleave", [x] + list(args), kwargs)


@register_op("Resize_as_", "ai.onnx")
def resize_as_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Resize_as_."""
    return record_op("Resize_as_", [x] + list(args), kwargs)


@register_op("Resize_as_sparse_", "ai.onnx")
def resize_as_sparse_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Resize_as_sparse_."""
    return record_op("Resize_as_sparse_", [x] + list(args), kwargs)


@register_op("Resolve_conj", "ai.onnx")
def resolve_conj(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Resolve_conj."""
    return record_op("Resolve_conj", [x] + list(args), kwargs)


@register_op("Resolve_neg", "ai.onnx")
def resolve_neg(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Resolve_neg."""
    return record_op("Resolve_neg", [x] + list(args), kwargs)


@register_op("Result_type", "ai.onnx")
def result_type(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Result_type."""
    return record_op("Result_type", [x] + list(args), kwargs)


@register_op("Rms_norm", "ai.onnx")
def rms_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rms_norm."""
    return record_op("Rms_norm", [x] + list(args), kwargs)


@register_op("Rnn_relu", "ai.onnx")
def rnn_relu(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rnn_relu."""
    return record_op("Rnn_relu", [x] + list(args), kwargs)


@register_op("Rnn_relu_cell", "ai.onnx")
def rnn_relu_cell(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rnn_relu_cell."""
    return record_op("Rnn_relu_cell", [x] + list(args), kwargs)


@register_op("Rnn_tanh", "ai.onnx")
def rnn_tanh(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rnn_tanh."""
    return record_op("Rnn_tanh", [x] + list(args), kwargs)


@register_op("Rnn_tanh_cell", "ai.onnx")
def rnn_tanh_cell(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rnn_tanh_cell."""
    return record_op("Rnn_tanh_cell", [x] + list(args), kwargs)


@register_op("Roll", "ai.onnx")
def roll(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Roll."""
    return record_op("Roll", [x] + list(args), kwargs)


@register_op("Rot90", "ai.onnx")
def rot90(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rot90."""
    return record_op("Rot90", [x] + list(args), kwargs)


@register_op("Round_", "ai.onnx")
def round_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Round_."""
    return record_op("Round_", [x] + list(args), kwargs)


@register_op("Row_indices_copy", "ai.onnx")
def row_indices_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Row_indices_copy."""
    return record_op("Row_indices_copy", [x] + list(args), kwargs)


@register_op("Row_stack", "ai.onnx")
def row_stack(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Row_stack."""
    return record_op("Row_stack", [x] + list(args), kwargs)


@register_op("Rrelu", "ai.onnx")
def rrelu(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rrelu."""
    return record_op("Rrelu", [x] + list(args), kwargs)


@register_op("Rrelu_", "ai.onnx")
def rrelu_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rrelu_."""
    return record_op("Rrelu_", [x] + list(args), kwargs)


@register_op("Rsqrt", "ai.onnx")
def rsqrt(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rsqrt."""
    return record_op("Rsqrt", [x] + list(args), kwargs)


@register_op("Rsqrt_", "ai.onnx")
def rsqrt_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rsqrt_."""
    return record_op("Rsqrt_", [x] + list(args), kwargs)


@register_op("Rsub", "ai.onnx")
def rsub(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rsub."""
    return record_op("Rsub", [x] + list(args), kwargs)


@register_op("Saddmm", "ai.onnx")
def saddmm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Saddmm."""
    return record_op("Saddmm", [x] + list(args), kwargs)


@register_op("Scalar_tensor", "ai.onnx")
def scalar_tensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scalar_tensor."""
    return record_op("Scalar_tensor", [x] + list(args), kwargs)


@register_op("Scatter_add", "ai.onnx")
def scatter_add(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scatter_add."""
    return record_op("Scatter_add", [x] + list(args), kwargs)


@register_op("Scatter_reduce", "ai.onnx")
def scatter_reduce(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scatter_reduce."""
    return record_op("Scatter_reduce", [x] + list(args), kwargs)


@register_op("Searchsorted", "ai.onnx")
def searchsorted(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Searchsorted."""
    return record_op("Searchsorted", [x] + list(args), kwargs)


@register_op("Select", "ai.onnx")
def select(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Select."""
    return record_op("Select", [x] + list(args), kwargs)


@register_op("Select_copy", "ai.onnx")
def select_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Select_copy."""
    return record_op("Select_copy", [x] + list(args), kwargs)


@register_op("Select_scatter", "ai.onnx")
def select_scatter(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Select_scatter."""
    return record_op("Select_scatter", [x] + list(args), kwargs)


@register_op("Selu_", "ai.onnx")
def selu_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Selu_."""
    return record_op("Selu_", [x] + list(args), kwargs)


@register_op("Sgn", "ai.onnx")
def sgn(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sgn."""
    return record_op("Sgn", [x] + list(args), kwargs)


@register_op("Sigmoid_", "ai.onnx")
def sigmoid_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sigmoid_."""
    return record_op("Sigmoid_", [x] + list(args), kwargs)


@register_op("Signbit", "ai.onnx")
def signbit(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Signbit."""
    return record_op("Signbit", [x] + list(args), kwargs)


@register_op("Sin_", "ai.onnx")
def sin_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sin_."""
    return record_op("Sin_", [x] + list(args), kwargs)


@register_op("Sinc", "ai.onnx")
def sinc(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sinc."""
    return record_op("Sinc", [x] + list(args), kwargs)


@register_op("Sinc_", "ai.onnx")
def sinc_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sinc_."""
    return record_op("Sinc_", [x] + list(args), kwargs)


@register_op("Sinh_", "ai.onnx")
def sinh_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sinh_."""
    return record_op("Sinh_", [x] + list(args), kwargs)


@register_op("Slice_copy", "ai.onnx")
def slice_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Slice_copy."""
    return record_op("Slice_copy", [x] + list(args), kwargs)


@register_op("Slice_inverse", "ai.onnx")
def slice_inverse(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Slice_inverse."""
    return record_op("Slice_inverse", [x] + list(args), kwargs)


@register_op("Slice_scatter", "ai.onnx")
def slice_scatter(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Slice_scatter."""
    return record_op("Slice_scatter", [x] + list(args), kwargs)


@register_op("Slogdet", "ai.onnx")
def slogdet(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Slogdet."""
    return record_op("Slogdet", [x] + list(args), kwargs)


@register_op("Smm", "ai.onnx")
def smm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Smm."""
    return record_op("Smm", [x] + list(args), kwargs)


@register_op("Sort", "ai.onnx")
def sort(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sort."""
    return record_op("Sort", [x] + list(args), kwargs)


@register_op("Sparse_bsc_tensor", "ai.onnx")
def sparse_bsc_tensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sparse_bsc_tensor."""
    return record_op("Sparse_bsc_tensor", [x] + list(args), kwargs)


@register_op("Sparse_bsr_tensor", "ai.onnx")
def sparse_bsr_tensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sparse_bsr_tensor."""
    return record_op("Sparse_bsr_tensor", [x] + list(args), kwargs)


@register_op("Sparse_compressed_tensor", "ai.onnx")
def sparse_compressed_tensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sparse_compressed_tensor."""
    return record_op("Sparse_compressed_tensor", [x] + list(args), kwargs)


@register_op("Sparse_coo_tensor", "ai.onnx")
def sparse_coo_tensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sparse_coo_tensor."""
    return record_op("Sparse_coo_tensor", [x] + list(args), kwargs)


@register_op("Sparse_csc_tensor", "ai.onnx")
def sparse_csc_tensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sparse_csc_tensor."""
    return record_op("Sparse_csc_tensor", [x] + list(args), kwargs)


@register_op("Sparse_csr_tensor", "ai.onnx")
def sparse_csr_tensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sparse_csr_tensor."""
    return record_op("Sparse_csr_tensor", [x] + list(args), kwargs)


@register_op("Split_copy", "ai.onnx")
def split_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Split_copy."""
    return record_op("Split_copy", [x] + list(args), kwargs)


@register_op("Split_with_sizes", "ai.onnx")
def split_with_sizes(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Split_with_sizes."""
    return record_op("Split_with_sizes", [x] + list(args), kwargs)


@register_op("Split_with_sizes_copy", "ai.onnx")
def split_with_sizes_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Split_with_sizes_copy."""
    return record_op("Split_with_sizes_copy", [x] + list(args), kwargs)


@register_op("Spmm", "ai.onnx")
def spmm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Spmm."""
    return record_op("Spmm", [x] + list(args), kwargs)


@register_op("Sqrt", "ai.onnx")
def sqrt(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sqrt."""
    return record_op("Sqrt", [x] + list(args), kwargs)


@register_op("Sqrt_", "ai.onnx")
def sqrt_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sqrt_."""
    return record_op("Sqrt_", [x] + list(args), kwargs)


@register_op("Square", "ai.onnx")
def square(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Square."""
    return record_op("Square", [x] + list(args), kwargs)


@register_op("Square_", "ai.onnx")
def square_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Square_."""
    return record_op("Square_", [x] + list(args), kwargs)


@register_op("Squeeze", "ai.onnx")
def squeeze(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Squeeze."""
    return record_op("Squeeze", [x] + list(args), kwargs)


@register_op("Squeeze_copy", "ai.onnx")
def squeeze_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Squeeze_copy."""
    return record_op("Squeeze_copy", [x] + list(args), kwargs)


@register_op("Sspaddmm", "ai.onnx")
def sspaddmm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sspaddmm."""
    return record_op("Sspaddmm", [x] + list(args), kwargs)


@register_op("Std", "ai.onnx")
def std(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Std."""
    return record_op("Std", [x] + list(args), kwargs)


@register_op("Std_mean", "ai.onnx")
def std_mean(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Std_mean."""
    return record_op("Std_mean", [x] + list(args), kwargs)


@register_op("Stft", "ai.onnx")
def stft(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Stft."""
    return record_op("Stft", [x] + list(args), kwargs)


@register_op("Subtract", "ai.onnx")
def subtract(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Subtract."""
    return record_op("Subtract", [x] + list(args), kwargs)


@register_op("Svd", "ai.onnx")
def svd(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Svd."""
    return record_op("Svd", [x] + list(args), kwargs)


@register_op("Swapaxes", "ai.onnx")
def swapaxes(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Swapaxes."""
    return record_op("Swapaxes", [x] + list(args), kwargs)


@register_op("Swapdims", "ai.onnx")
def swapdims(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Swapdims."""
    return record_op("Swapdims", [x] + list(args), kwargs)


@register_op("Sym_constrain_range", "ai.onnx")
def sym_constrain_range(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sym_constrain_range."""
    return record_op("Sym_constrain_range", [x] + list(args), kwargs)


@register_op("Sym_constrain_range_for_size", "ai.onnx")
def sym_constrain_range_for_size(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sym_constrain_range_for_size."""
    return record_op("Sym_constrain_range_for_size", [x] + list(args), kwargs)


@register_op("T", "ai.onnx")
def t(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute T."""
    return record_op("T", [x] + list(args), kwargs)


@register_op("T_copy", "ai.onnx")
def t_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute T_copy."""
    return record_op("T_copy", [x] + list(args), kwargs)


@register_op("Take", "ai.onnx")
def take(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Take."""
    return record_op("Take", [x] + list(args), kwargs)


@register_op("Take_along_dim", "ai.onnx")
def take_along_dim(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Take_along_dim."""
    return record_op("Take_along_dim", [x] + list(args), kwargs)


@register_op("Tan_", "ai.onnx")
def tan_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Tan_."""
    return record_op("Tan_", [x] + list(args), kwargs)


@register_op("Tanh_", "ai.onnx")
def tanh_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Tanh_."""
    return record_op("Tanh_", [x] + list(args), kwargs)


@register_op("Tensor_split", "ai.onnx")
def tensor_split(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Tensor_split."""
    return record_op("Tensor_split", [x] + list(args), kwargs)


@register_op("Tensordot", "ai.onnx")
def tensordot(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Tensordot."""
    return record_op("Tensordot", [x] + list(args), kwargs)


@register_op("Threshold", "ai.onnx")
def threshold(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Threshold."""
    return record_op("Threshold", [x] + list(args), kwargs)


@register_op("Threshold_", "ai.onnx")
def threshold_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Threshold_."""
    return record_op("Threshold_", [x] + list(args), kwargs)


@register_op("Transpose_copy", "ai.onnx")
def transpose_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Transpose_copy."""
    return record_op("Transpose_copy", [x] + list(args), kwargs)


@register_op("Trapezoid", "ai.onnx")
def trapezoid(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Trapezoid."""
    return record_op("Trapezoid", [x] + list(args), kwargs)


@register_op("Trapz", "ai.onnx")
def trapz(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Trapz."""
    return record_op("Trapz", [x] + list(args), kwargs)


@register_op("Triangular_solve", "ai.onnx")
def triangular_solve(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Triangular_solve."""
    return record_op("Triangular_solve", [x] + list(args), kwargs)


@register_op("Tril", "ai.onnx")
def tril(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Tril."""
    return record_op("Tril", [x] + list(args), kwargs)


@register_op("Tril_indices", "ai.onnx")
def tril_indices(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Tril_indices."""
    return record_op("Tril_indices", [x] + list(args), kwargs)


@register_op("Triplet_margin_loss", "ai.onnx")
def triplet_margin_loss(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Triplet_margin_loss."""
    return record_op("Triplet_margin_loss", [x] + list(args), kwargs)


@register_op("Triu", "ai.onnx")
def triu(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Triu."""
    return record_op("Triu", [x] + list(args), kwargs)


@register_op("Triu_indices", "ai.onnx")
def triu_indices(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Triu_indices."""
    return record_op("Triu_indices", [x] + list(args), kwargs)


@register_op("True_divide", "ai.onnx")
def true_divide(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute True_divide."""
    return record_op("True_divide", [x] + list(args), kwargs)


@register_op("Trunc", "ai.onnx")
def trunc(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Trunc."""
    return record_op("Trunc", [x] + list(args), kwargs)


@register_op("Trunc_", "ai.onnx")
def trunc_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Trunc_."""
    return record_op("Trunc_", [x] + list(args), kwargs)


@register_op("Unbind", "ai.onnx")
def unbind(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Unbind."""
    return record_op("Unbind", [x] + list(args), kwargs)


@register_op("Unbind_copy", "ai.onnx")
def unbind_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Unbind_copy."""
    return record_op("Unbind_copy", [x] + list(args), kwargs)


@register_op("Unflatten", "ai.onnx")
def unflatten(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Unflatten."""
    return record_op("Unflatten", [x] + list(args), kwargs)


@register_op("Unfold_copy", "ai.onnx")
def unfold_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Unfold_copy."""
    return record_op("Unfold_copy", [x] + list(args), kwargs)


@register_op("Unique_consecutive", "ai.onnx")
def unique_consecutive(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Unique_consecutive."""
    return record_op("Unique_consecutive", [x] + list(args), kwargs)


@register_op("Unsafe_chunk", "ai.onnx")
def unsafe_chunk(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Unsafe_chunk."""
    return record_op("Unsafe_chunk", [x] + list(args), kwargs)


@register_op("Unsafe_split", "ai.onnx")
def unsafe_split(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Unsafe_split."""
    return record_op("Unsafe_split", [x] + list(args), kwargs)


@register_op("Unsafe_split_with_sizes", "ai.onnx")
def unsafe_split_with_sizes(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Unsafe_split_with_sizes."""
    return record_op("Unsafe_split_with_sizes", [x] + list(args), kwargs)


@register_op("Unsqueeze", "ai.onnx")
def unsqueeze(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Unsqueeze."""
    return record_op("Unsqueeze", [x] + list(args), kwargs)


@register_op("Unsqueeze_copy", "ai.onnx")
def unsqueeze_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Unsqueeze_copy."""
    return record_op("Unsqueeze_copy", [x] + list(args), kwargs)


@register_op("Values_copy", "ai.onnx")
def values_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Values_copy."""
    return record_op("Values_copy", [x] + list(args), kwargs)


@register_op("Vander", "ai.onnx")
def vander(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Vander."""
    return record_op("Vander", [x] + list(args), kwargs)


@register_op("Var", "ai.onnx")
def var(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Var."""
    return record_op("Var", [x] + list(args), kwargs)


@register_op("Var_mean", "ai.onnx")
def var_mean(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Var_mean."""
    return record_op("Var_mean", [x] + list(args), kwargs)


@register_op("Vdot", "ai.onnx")
def vdot(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Vdot."""
    return record_op("Vdot", [x] + list(args), kwargs)


@register_op("View_as_complex", "ai.onnx")
def view_as_complex(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute View_as_complex."""
    return record_op("View_as_complex", [x] + list(args), kwargs)


@register_op("View_as_complex_copy", "ai.onnx")
def view_as_complex_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute View_as_complex_copy."""
    return record_op("View_as_complex_copy", [x] + list(args), kwargs)


@register_op("View_as_real", "ai.onnx")
def view_as_real(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute View_as_real."""
    return record_op("View_as_real", [x] + list(args), kwargs)


@register_op("View_as_real_copy", "ai.onnx")
def view_as_real_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute View_as_real_copy."""
    return record_op("View_as_real_copy", [x] + list(args), kwargs)


@register_op("View_copy", "ai.onnx")
def view_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute View_copy."""
    return record_op("View_copy", [x] + list(args), kwargs)


@register_op("Vsplit", "ai.onnx")
def vsplit(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Vsplit."""
    return record_op("Vsplit", [x] + list(args), kwargs)


@register_op("Vstack", "ai.onnx")
def vstack(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Vstack."""
    return record_op("Vstack", [x] + list(args), kwargs)


@register_op("Xlogy", "ai.onnx")
def xlogy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Xlogy."""
    return record_op("Xlogy", [x] + list(args), kwargs)


@register_op("Xlogy_", "ai.onnx")
def xlogy_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Xlogy_."""
    return record_op("Xlogy_", [x] + list(args), kwargs)


@register_op("Zero_", "ai.onnx")
def zero_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Zero_."""
    return record_op("Zero_", [x] + list(args), kwargs)


@register_op("Zeros_like", "ai.onnx")
def zeros_like(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Zeros_like."""
    return record_op("Zeros_like", [x] + list(args), kwargs)


@register_op("Bfloat16", "ai.onnx")
def bfloat16(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bfloat16."""
    return record_op("Bfloat16", [x] + list(args), kwargs)


@register_op("Bit", "ai.onnx")
def bit(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bit."""
    return record_op("Bit", [x] + list(args), kwargs)


@register_op("Bits16", "ai.onnx")
def bits16(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bits16."""
    return record_op("Bits16", [x] + list(args), kwargs)


@register_op("Bits1x8", "ai.onnx")
def bits1x8(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bits1x8."""
    return record_op("Bits1x8", [x] + list(args), kwargs)


@register_op("Bits2x4", "ai.onnx")
def bits2x4(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bits2x4."""
    return record_op("Bits2x4", [x] + list(args), kwargs)


@register_op("Bits4x2", "ai.onnx")
def bits4x2(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bits4x2."""
    return record_op("Bits4x2", [x] + list(args), kwargs)


@register_op("Bits8", "ai.onnx")
def bits8(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bits8."""
    return record_op("Bits8", [x] + list(args), kwargs)


@register_op("Cdouble", "ai.onnx")
def cdouble(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cdouble."""
    return record_op("Cdouble", [x] + list(args), kwargs)


@register_op("Cfloat", "ai.onnx")
def cfloat(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cfloat."""
    return record_op("Cfloat", [x] + list(args), kwargs)


@register_op("Chalf", "ai.onnx")
def chalf(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Chalf."""
    return record_op("Chalf", [x] + list(args), kwargs)


@register_op("Complex128", "ai.onnx")
def complex128(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Complex128."""
    return record_op("Complex128", [x] + list(args), kwargs)


@register_op("Complex32", "ai.onnx")
def complex32(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Complex32."""
    return record_op("Complex32", [x] + list(args), kwargs)


@register_op("Complex64", "ai.onnx")
def complex64(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Complex64."""
    return record_op("Complex64", [x] + list(args), kwargs)


@register_op("Double", "ai.onnx")
def double(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Double."""
    return record_op("Double", [x] + list(args), kwargs)


@register_op("Float", "ai.onnx")
def float(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Float."""
    return record_op("Float", [x] + list(args), kwargs)


@register_op("Float16", "ai.onnx")
def float16(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Float16."""
    return record_op("Float16", [x] + list(args), kwargs)


@register_op("Float4_e2m1fn_x2", "ai.onnx")
def float4_e2m1fn_x2(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Float4_e2m1fn_x2."""
    return record_op("Float4_e2m1fn_x2", [x] + list(args), kwargs)


@register_op("Float8_e4m3fn", "ai.onnx")
def float8_e4m3fn(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Float8_e4m3fn."""
    return record_op("Float8_e4m3fn", [x] + list(args), kwargs)


@register_op("Float8_e4m3fnuz", "ai.onnx")
def float8_e4m3fnuz(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Float8_e4m3fnuz."""
    return record_op("Float8_e4m3fnuz", [x] + list(args), kwargs)


@register_op("Float8_e5m2", "ai.onnx")
def float8_e5m2(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Float8_e5m2."""
    return record_op("Float8_e5m2", [x] + list(args), kwargs)


@register_op("Float8_e5m2fnuz", "ai.onnx")
def float8_e5m2fnuz(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Float8_e5m2fnuz."""
    return record_op("Float8_e5m2fnuz", [x] + list(args), kwargs)


@register_op("Float8_e8m0fnu", "ai.onnx")
def float8_e8m0fnu(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Float8_e8m0fnu."""
    return record_op("Float8_e8m0fnu", [x] + list(args), kwargs)


@register_op("Half", "ai.onnx")
def half(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Half."""
    return record_op("Half", [x] + list(args), kwargs)


@register_op("Int", "ai.onnx")
def int(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Int."""
    return record_op("Int", [x] + list(args), kwargs)


@register_op("Int1", "ai.onnx")
def int1(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Int1."""
    return record_op("Int1", [x] + list(args), kwargs)


@register_op("Int16", "ai.onnx")
def int16(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Int16."""
    return record_op("Int16", [x] + list(args), kwargs)


@register_op("Int2", "ai.onnx")
def int2(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Int2."""
    return record_op("Int2", [x] + list(args), kwargs)


@register_op("Int3", "ai.onnx")
def int3(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Int3."""
    return record_op("Int3", [x] + list(args), kwargs)


@register_op("Int4", "ai.onnx")
def int4(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Int4."""
    return record_op("Int4", [x] + list(args), kwargs)


@register_op("Int5", "ai.onnx")
def int5(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Int5."""
    return record_op("Int5", [x] + list(args), kwargs)


@register_op("Int6", "ai.onnx")
def int6(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Int6."""
    return record_op("Int6", [x] + list(args), kwargs)


@register_op("Int7", "ai.onnx")
def int7(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Int7."""
    return record_op("Int7", [x] + list(args), kwargs)


@register_op("Int8", "ai.onnx")
def int8(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Int8."""
    return record_op("Int8", [x] + list(args), kwargs)


@register_op("Long", "ai.onnx")
def long(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Long."""
    return record_op("Long", [x] + list(args), kwargs)


@register_op("Qint32", "ai.onnx")
def qint32(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Qint32."""
    return record_op("Qint32", [x] + list(args), kwargs)


@register_op("Qint8", "ai.onnx")
def qint8(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Qint8."""
    return record_op("Qint8", [x] + list(args), kwargs)


@register_op("Quint2x4", "ai.onnx")
def quint2x4(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quint2x4."""
    return record_op("Quint2x4", [x] + list(args), kwargs)


@register_op("Quint4x2", "ai.onnx")
def quint4x2(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quint4x2."""
    return record_op("Quint4x2", [x] + list(args), kwargs)


@register_op("Quint8", "ai.onnx")
def quint8(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quint8."""
    return record_op("Quint8", [x] + list(args), kwargs)


@register_op("Short", "ai.onnx")
def short(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Short."""
    return record_op("Short", [x] + list(args), kwargs)


@register_op("Uint1", "ai.onnx")
def uint1(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Uint1."""
    return record_op("Uint1", [x] + list(args), kwargs)


@register_op("Uint16", "ai.onnx")
def uint16(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Uint16."""
    return record_op("Uint16", [x] + list(args), kwargs)


@register_op("Uint2", "ai.onnx")
def uint2(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Uint2."""
    return record_op("Uint2", [x] + list(args), kwargs)


@register_op("Uint3", "ai.onnx")
def uint3(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Uint3."""
    return record_op("Uint3", [x] + list(args), kwargs)


@register_op("Uint32", "ai.onnx")
def uint32(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Uint32."""
    return record_op("Uint32", [x] + list(args), kwargs)


@register_op("Uint4", "ai.onnx")
def uint4(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Uint4."""
    return record_op("Uint4", [x] + list(args), kwargs)


@register_op("Uint5", "ai.onnx")
def uint5(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Uint5."""
    return record_op("Uint5", [x] + list(args), kwargs)


@register_op("Uint6", "ai.onnx")
def uint6(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Uint6."""
    return record_op("Uint6", [x] + list(args), kwargs)


@register_op("Uint64", "ai.onnx")
def uint64(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Uint64."""
    return record_op("Uint64", [x] + list(args), kwargs)


@register_op("Uint7", "ai.onnx")
def uint7(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Uint7."""
    return record_op("Uint7", [x] + list(args), kwargs)


@register_op("Uint8", "ai.onnx")
def uint8(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Uint8."""
    return record_op("Uint8", [x] + list(args), kwargs)
