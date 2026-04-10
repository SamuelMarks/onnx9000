"""Auto-generated core ops for torch compliance."""

from typing import Any

from onnx9000.core.ir import Tensor
from onnx9000.core.ops import record_op
from onnx9000.core.registry import register_op


@register_op("ai.onnx", "Boolstorage")
def BoolStorage(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Boolstorage."""
    return record_op("Boolstorage", [x] + list(args), kwargs)


@register_op("ai.onnx", "Booltensor")
def BoolTensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Booltensor."""
    return record_op("Booltensor", [x] + list(args), kwargs)


@register_op("ai.onnx", "Bytestorage")
def ByteStorage(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bytestorage."""
    return record_op("Bytestorage", [x] + list(args), kwargs)


@register_op("ai.onnx", "Bytetensor")
def ByteTensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bytetensor."""
    return record_op("Bytetensor", [x] + list(args), kwargs)


@register_op("ai.onnx", "Charstorage")
def CharStorage(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Charstorage."""
    return record_op("Charstorage", [x] + list(args), kwargs)


@register_op("ai.onnx", "Chartensor")
def CharTensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Chartensor."""
    return record_op("Chartensor", [x] + list(args), kwargs)


@register_op("ai.onnx", "Doublestorage")
def DoubleStorage(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Doublestorage."""
    return record_op("Doublestorage", [x] + list(args), kwargs)


@register_op("ai.onnx", "Doubletensor")
def DoubleTensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Doubletensor."""
    return record_op("Doubletensor", [x] + list(args), kwargs)


@register_op("ai.onnx", "Floatstorage")
def FloatStorage(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Floatstorage."""
    return record_op("Floatstorage", [x] + list(args), kwargs)


@register_op("ai.onnx", "Floattensor")
def FloatTensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Floattensor."""
    return record_op("Floattensor", [x] + list(args), kwargs)


@register_op("ai.onnx", "Gradscaler")
def GradScaler(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Gradscaler."""
    return record_op("Gradscaler", [x] + list(args), kwargs)


@register_op("ai.onnx", "Intstorage")
def IntStorage(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Intstorage."""
    return record_op("Intstorage", [x] + list(args), kwargs)


@register_op("ai.onnx", "Inttensor")
def IntTensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Inttensor."""
    return record_op("Inttensor", [x] + list(args), kwargs)


@register_op("ai.onnx", "Longstorage")
def LongStorage(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Longstorage."""
    return record_op("Longstorage", [x] + list(args), kwargs)


@register_op("ai.onnx", "Longtensor")
def LongTensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Longtensor."""
    return record_op("Longtensor", [x] + list(args), kwargs)


@register_op("ai.onnx", "Shortstorage")
def ShortStorage(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Shortstorage."""
    return record_op("Shortstorage", [x] + list(args), kwargs)


@register_op("ai.onnx", "Shorttensor")
def ShortTensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Shorttensor."""
    return record_op("Shorttensor", [x] + list(args), kwargs)


@register_op("ai.onnx", "Symbool")
def SymBool(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Symbool."""
    return record_op("Symbool", [x] + list(args), kwargs)


@register_op("ai.onnx", "Symfloat")
def SymFloat(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Symfloat."""
    return record_op("Symfloat", [x] + list(args), kwargs)


@register_op("ai.onnx", "Symint")
def SymInt(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Symint."""
    return record_op("Symint", [x] + list(args), kwargs)


@register_op("ai.onnx", "Typedstorage")
def TypedStorage(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Typedstorage."""
    return record_op("Typedstorage", [x] + list(args), kwargs)


@register_op("ai.onnx", "Untypedstorage")
def UntypedStorage(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Untypedstorage."""
    return record_op("Untypedstorage", [x] + list(args), kwargs)


@register_op("ai.onnx", "Are_deterministic_algorithms_enabled")
def are_deterministic_algorithms_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Are_deterministic_algorithms_enabled."""
    return record_op("Are_deterministic_algorithms_enabled", [x] + list(args), kwargs)


@register_op("ai.onnx", "Autocast")
def autocast(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Autocast."""
    return record_op("Autocast", [x] + list(args), kwargs)


@register_op("ai.onnx", "Chunk")
def chunk(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Chunk."""
    return record_op("Chunk", [x] + list(args), kwargs)


@register_op("ai.onnx", "Compile")
def compile(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Compile."""
    return record_op("Compile", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cond")
def cond(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cond."""
    return record_op("Cond", [x] + list(args), kwargs)


@register_op("ai.onnx", "Enable_grad")
def enable_grad(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Enable_grad."""
    return record_op("Enable_grad", [x] + list(args), kwargs)


@register_op("ai.onnx", "Export_additionalinputs")
def export_AdditionalInputs(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_additionalinputs."""
    return record_op("Export_additionalinputs", [x] + list(args), kwargs)


@register_op("ai.onnx", "Export_constraint")
def export_Constraint(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_constraint."""
    return record_op("Export_constraint", [x] + list(args), kwargs)


@register_op("ai.onnx", "Export_customdecomptable")
def export_CustomDecompTable(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_customdecomptable."""
    return record_op("Export_customdecomptable", [x] + list(args), kwargs)


@register_op("ai.onnx", "Export_default_decompositions")
def export_default_decompositions(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_default_decompositions."""
    return record_op("Export_default_decompositions", [x] + list(args), kwargs)


@register_op("ai.onnx", "Export_dim")
def export_Dim(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_dim."""
    return record_op("Export_dim", [x] + list(args), kwargs)


@register_op("ai.onnx", "Export_dims")
def export_dims(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_dims."""
    return record_op("Export_dims", [x] + list(args), kwargs)


@register_op("ai.onnx", "Export_draft_export")
def export_draft_export(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_draft_export."""
    return record_op("Export_draft_export", [x] + list(args), kwargs)


@register_op("ai.onnx", "Export_export")
def export_export(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_export."""
    return record_op("Export_export", [x] + list(args), kwargs)


@register_op("ai.onnx", "Export_exportbackwardsignature")
def export_ExportBackwardSignature(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_exportbackwardsignature."""
    return record_op("Export_exportbackwardsignature", [x] + list(args), kwargs)


@register_op("ai.onnx", "Export_exportedprogram")
def export_ExportedProgram(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_exportedprogram."""
    return record_op("Export_exportedprogram", [x] + list(args), kwargs)


@register_op("ai.onnx", "Export_exportgraphsignature")
def export_ExportGraphSignature(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_exportgraphsignature."""
    return record_op("Export_exportgraphsignature", [x] + list(args), kwargs)


@register_op("ai.onnx", "Export_flatargsadapter")
def export_FlatArgsAdapter(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_flatargsadapter."""
    return record_op("Export_flatargsadapter", [x] + list(args), kwargs)


@register_op("ai.onnx", "Export_load")
def export_load(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_load."""
    return record_op("Export_load", [x] + list(args), kwargs)


@register_op("ai.onnx", "Export_modulecallentry")
def export_ModuleCallEntry(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_modulecallentry."""
    return record_op("Export_modulecallentry", [x] + list(args), kwargs)


@register_op("ai.onnx", "Export_modulecallsignature")
def export_ModuleCallSignature(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_modulecallsignature."""
    return record_op("Export_modulecallsignature", [x] + list(args), kwargs)


@register_op("ai.onnx", "Export_register_dataclass")
def export_register_dataclass(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_register_dataclass."""
    return record_op("Export_register_dataclass", [x] + list(args), kwargs)


@register_op("ai.onnx", "Export_save")
def export_save(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_save."""
    return record_op("Export_save", [x] + list(args), kwargs)


@register_op("ai.onnx", "Export_shapescollection")
def export_ShapesCollection(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_shapescollection."""
    return record_op("Export_shapescollection", [x] + list(args), kwargs)


@register_op("ai.onnx", "Export_unflatten")
def export_unflatten(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_unflatten."""
    return record_op("Export_unflatten", [x] + list(args), kwargs)


@register_op("ai.onnx", "Export_unflattenedmodule")
def export_UnflattenedModule(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Export_unflattenedmodule."""
    return record_op("Export_unflattenedmodule", [x] + list(args), kwargs)


@register_op("ai.onnx", "Get_default_device")
def get_default_device(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_default_device."""
    return record_op("Get_default_device", [x] + list(args), kwargs)


@register_op("ai.onnx", "Get_deterministic_debug_mode")
def get_deterministic_debug_mode(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_deterministic_debug_mode."""
    return record_op("Get_deterministic_debug_mode", [x] + list(args), kwargs)


@register_op("ai.onnx", "Get_device_module")
def get_device_module(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_device_module."""
    return record_op("Get_device_module", [x] + list(args), kwargs)


@register_op("ai.onnx", "Get_float32_matmul_precision")
def get_float32_matmul_precision(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_float32_matmul_precision."""
    return record_op("Get_float32_matmul_precision", [x] + list(args), kwargs)


@register_op("ai.onnx", "Get_rng_state")
def get_rng_state(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_rng_state."""
    return record_op("Get_rng_state", [x] + list(args), kwargs)


@register_op("ai.onnx", "Inference_mode")
def inference_mode(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Inference_mode."""
    return record_op("Inference_mode", [x] + list(args), kwargs)


@register_op("ai.onnx", "Initial_seed")
def initial_seed(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Initial_seed."""
    return record_op("Initial_seed", [x] + list(args), kwargs)


@register_op("ai.onnx", "Is_deterministic_algorithms_warn_only_enabled")
def is_deterministic_algorithms_warn_only_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_deterministic_algorithms_warn_only_enabled."""
    return record_op("Is_deterministic_algorithms_warn_only_enabled", [x] + list(args), kwargs)


@register_op("ai.onnx", "Is_storage")
def is_storage(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_storage."""
    return record_op("Is_storage", [x] + list(args), kwargs)


@register_op("ai.onnx", "Is_tensor")
def is_tensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_tensor."""
    return record_op("Is_tensor", [x] + list(args), kwargs)


@register_op("ai.onnx", "Is_warn_always_enabled")
def is_warn_always_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_warn_always_enabled."""
    return record_op("Is_warn_always_enabled", [x] + list(args), kwargs)


@register_op("ai.onnx", "Load")
def load(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Load."""
    return record_op("Load", [x] + list(args), kwargs)


@register_op("ai.onnx", "Lobpcg")
def lobpcg(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Lobpcg."""
    return record_op("Lobpcg", [x] + list(args), kwargs)


@register_op("ai.onnx", "Manual_seed")
def manual_seed(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Manual_seed."""
    return record_op("Manual_seed", [x] + list(args), kwargs)


@register_op("ai.onnx", "No_grad")
def no_grad(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute No_grad."""
    return record_op("No_grad", [x] + list(args), kwargs)


@register_op("ai.onnx", "Rand")
def rand(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rand."""
    return record_op("Rand", [x] + list(args), kwargs)


@register_op("ai.onnx", "Save")
def save(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Save."""
    return record_op("Save", [x] + list(args), kwargs)


@register_op("ai.onnx", "Seed")
def seed(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Seed."""
    return record_op("Seed", [x] + list(args), kwargs)


@register_op("ai.onnx", "Set_default_device")
def set_default_device(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_default_device."""
    return record_op("Set_default_device", [x] + list(args), kwargs)


@register_op("ai.onnx", "Set_default_tensor_type")
def set_default_tensor_type(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_default_tensor_type."""
    return record_op("Set_default_tensor_type", [x] + list(args), kwargs)


@register_op("ai.onnx", "Set_deterministic_debug_mode")
def set_deterministic_debug_mode(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_deterministic_debug_mode."""
    return record_op("Set_deterministic_debug_mode", [x] + list(args), kwargs)


@register_op("ai.onnx", "Set_float32_matmul_precision")
def set_float32_matmul_precision(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_float32_matmul_precision."""
    return record_op("Set_float32_matmul_precision", [x] + list(args), kwargs)


@register_op("ai.onnx", "Set_printoptions")
def set_printoptions(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_printoptions."""
    return record_op("Set_printoptions", [x] + list(args), kwargs)


@register_op("ai.onnx", "Set_rng_state")
def set_rng_state(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_rng_state."""
    return record_op("Set_rng_state", [x] + list(args), kwargs)


@register_op("ai.onnx", "Set_warn_always")
def set_warn_always(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_warn_always."""
    return record_op("Set_warn_always", [x] + list(args), kwargs)


@register_op("ai.onnx", "Split")
def split(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Split."""
    return record_op("Split", [x] + list(args), kwargs)


@register_op("ai.onnx", "Stack")
def stack(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Stack."""
    return record_op("Stack", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sym_float")
def sym_float(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sym_float."""
    return record_op("Sym_float", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sym_fresh_size")
def sym_fresh_size(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sym_fresh_size."""
    return record_op("Sym_fresh_size", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sym_int")
def sym_int(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sym_int."""
    return record_op("Sym_int", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sym_ite")
def sym_ite(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sym_ite."""
    return record_op("Sym_ite", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sym_max")
def sym_max(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sym_max."""
    return record_op("Sym_max", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sym_min")
def sym_min(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sym_min."""
    return record_op("Sym_min", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sym_not")
def sym_not(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sym_not."""
    return record_op("Sym_not", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sym_sum")
def sym_sum(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sym_sum."""
    return record_op("Sym_sum", [x] + list(args), kwargs)


@register_op("ai.onnx", "Typename")
def typename(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Typename."""
    return record_op("Typename", [x] + list(args), kwargs)


@register_op("ai.onnx", "Unravel_index")
def unravel_index(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Unravel_index."""
    return record_op("Unravel_index", [x] + list(args), kwargs)


@register_op("ai.onnx", "Use_deterministic_algorithms")
def use_deterministic_algorithms(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Use_deterministic_algorithms."""
    return record_op("Use_deterministic_algorithms", [x] + list(args), kwargs)


@register_op("ai.onnx", "Vmap")
def vmap(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Vmap."""
    return record_op("Vmap", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sym_sqrt")
def sym_sqrt(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sym_sqrt."""
    return record_op("Sym_sqrt", [x] + list(args), kwargs)


@register_op("ai.onnx", "Avg")
def AVG(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Avg."""
    return record_op("Avg", [x] + list(args), kwargs)


@register_op("ai.onnx", "Acceleratorerror")
def AcceleratorError(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Acceleratorerror."""
    return record_op("Acceleratorerror", [x] + list(args), kwargs)


@register_op("ai.onnx", "Aggregationtype")
def AggregationType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Aggregationtype."""
    return record_op("Aggregationtype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Aliasdb")
def AliasDb(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Aliasdb."""
    return record_op("Aliasdb", [x] + list(args), kwargs)


@register_op("ai.onnx", "Anytype")
def AnyType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Anytype."""
    return record_op("Anytype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Argument")
def Argument(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Argument."""
    return record_op("Argument", [x] + list(args), kwargs)


@register_op("ai.onnx", "Argumentspec")
def ArgumentSpec(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Argumentspec."""
    return record_op("Argumentspec", [x] + list(args), kwargs)


@register_op("ai.onnx", "Awaittype")
def AwaitType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Awaittype."""
    return record_op("Awaittype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Benchmarkconfig")
def BenchmarkConfig(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Benchmarkconfig."""
    return record_op("Benchmarkconfig", [x] + list(args), kwargs)


@register_op("ai.onnx", "Benchmarkexecutionstats")
def BenchmarkExecutionStats(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Benchmarkexecutionstats."""
    return record_op("Benchmarkexecutionstats", [x] + list(args), kwargs)


@register_op("ai.onnx", "Block")
def Block(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Block."""
    return record_op("Block", [x] + list(args), kwargs)


@register_op("ai.onnx", "Booltype")
def BoolType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Booltype."""
    return record_op("Booltype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Bufferdict")
def BufferDict(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bufferdict."""
    return record_op("Bufferdict", [x] + list(args), kwargs)


@register_op("ai.onnx", "Callstack")
def CallStack(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Callstack."""
    return record_op("Callstack", [x] + list(args), kwargs)


@register_op("ai.onnx", "Capsule")
def Capsule(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Capsule."""
    return record_op("Capsule", [x] + list(args), kwargs)


@register_op("ai.onnx", "Classtype")
def ClassType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Classtype."""
    return record_op("Classtype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Code")
def Code(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Code."""
    return record_op("Code", [x] + list(args), kwargs)


@register_op("ai.onnx", "Compilationunit")
def CompilationUnit(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Compilationunit."""
    return record_op("Compilationunit", [x] + list(args), kwargs)


@register_op("ai.onnx", "Completeargumentspec")
def CompleteArgumentSpec(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Completeargumentspec."""
    return record_op("Completeargumentspec", [x] + list(args), kwargs)


@register_op("ai.onnx", "Complextype")
def ComplexType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Complextype."""
    return record_op("Complextype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Concretemoduletype")
def ConcreteModuleType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Concretemoduletype."""
    return record_op("Concretemoduletype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Concretemoduletypebuilder")
def ConcreteModuleTypeBuilder(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Concretemoduletypebuilder."""
    return record_op("Concretemoduletypebuilder", [x] + list(args), kwargs)


@register_op("ai.onnx", "Deepcopymemotable")
def DeepCopyMemoTable(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Deepcopymemotable."""
    return record_op("Deepcopymemotable", [x] + list(args), kwargs)


@register_op("ai.onnx", "Deserializationstoragecontext")
def DeserializationStorageContext(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Deserializationstoragecontext."""
    return record_op("Deserializationstoragecontext", [x] + list(args), kwargs)


@register_op("ai.onnx", "Deviceobjtype")
def DeviceObjType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Deviceobjtype."""
    return record_op("Deviceobjtype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Dicttype")
def DictType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Dicttype."""
    return record_op("Dicttype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Disabletorchfunction")
def DisableTorchFunction(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Disabletorchfunction."""
    return record_op("Disabletorchfunction", [x] + list(args), kwargs)


@register_op("ai.onnx", "Disabletorchfunctionsubclass")
def DisableTorchFunctionSubclass(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Disabletorchfunctionsubclass."""
    return record_op("Disabletorchfunctionsubclass", [x] + list(args), kwargs)


@register_op("ai.onnx", "Dispatchkey")
def DispatchKey(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Dispatchkey."""
    return record_op("Dispatchkey", [x] + list(args), kwargs)


@register_op("ai.onnx", "Dispatchkeyset")
def DispatchKeySet(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Dispatchkeyset."""
    return record_op("Dispatchkeyset", [x] + list(args), kwargs)


@register_op("ai.onnx", "Enumtype")
def EnumType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Enumtype."""
    return record_op("Enumtype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Errorreport")
def ErrorReport(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Errorreport."""
    return record_op("Errorreport", [x] + list(args), kwargs)


@register_op("ai.onnx", "Event")
def Event(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Event."""
    return record_op("Event", [x] + list(args), kwargs)


@register_op("ai.onnx", "Excludedispatchkeyguard")
def ExcludeDispatchKeyGuard(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Excludedispatchkeyguard."""
    return record_op("Excludedispatchkeyguard", [x] + list(args), kwargs)


@register_op("ai.onnx", "Executionplan")
def ExecutionPlan(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Executionplan."""
    return record_op("Executionplan", [x] + list(args), kwargs)


@register_op("ai.onnx", "Fatalerror")
def FatalError(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fatalerror."""
    return record_op("Fatalerror", [x] + list(args), kwargs)


@register_op("ai.onnx", "Filecheck")
def FileCheck(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Filecheck."""
    return record_op("Filecheck", [x] + list(args), kwargs)


@register_op("ai.onnx", "Floattype")
def FloatType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Floattype."""
    return record_op("Floattype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Functionschema")
def FunctionSchema(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Functionschema."""
    return record_op("Functionschema", [x] + list(args), kwargs)


@register_op("ai.onnx", "Future")
def Future(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Future."""
    return record_op("Future", [x] + list(args), kwargs)


@register_op("ai.onnx", "Futuretype")
def FutureType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Futuretype."""
    return record_op("Futuretype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Generator")
def Generator(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Generator."""
    return record_op("Generator", [x] + list(args), kwargs)


@register_op("ai.onnx", "Gradient")
def Gradient(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Gradient."""
    return record_op("Gradient", [x] + list(args), kwargs)


@register_op("ai.onnx", "Graph")
def Graph(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Graph."""
    return record_op("Graph", [x] + list(args), kwargs)


@register_op("ai.onnx", "Graphexecutorstate")
def GraphExecutorState(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Graphexecutorstate."""
    return record_op("Graphexecutorstate", [x] + list(args), kwargs)


@register_op("ai.onnx", "Iodescriptor")
def IODescriptor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Iodescriptor."""
    return record_op("Iodescriptor", [x] + list(args), kwargs)


@register_op("ai.onnx", "Inferredtype")
def InferredType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Inferredtype."""
    return record_op("Inferredtype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Inttype")
def IntType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Inttype."""
    return record_op("Inttype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Interfacetype")
def InterfaceType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Interfacetype."""
    return record_op("Interfacetype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Jitexception")
def JITException(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Jitexception."""
    return record_op("Jitexception", [x] + list(args), kwargs)


@register_op("ai.onnx", "Listtype")
def ListType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Listtype."""
    return record_op("Listtype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Litescriptmodule")
def LiteScriptModule(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Litescriptmodule."""
    return record_op("Litescriptmodule", [x] + list(args), kwargs)


@register_op("ai.onnx", "Lockinglogger")
def LockingLogger(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Lockinglogger."""
    return record_op("Lockinglogger", [x] + list(args), kwargs)


@register_op("ai.onnx", "Moduledict")
def ModuleDict(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Moduledict."""
    return record_op("Moduledict", [x] + list(args), kwargs)


@register_op("ai.onnx", "Node")
def Node(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Node."""
    return record_op("Node", [x] + list(args), kwargs)


@register_op("ai.onnx", "Nonetype")
def NoneType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Nonetype."""
    return record_op("Nonetype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Nooplogger")
def NoopLogger(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Nooplogger."""
    return record_op("Nooplogger", [x] + list(args), kwargs)


@register_op("ai.onnx", "Numbertype")
def NumberType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Numbertype."""
    return record_op("Numbertype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Operatorinfo")
def OperatorInfo(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Operatorinfo."""
    return record_op("Operatorinfo", [x] + list(args), kwargs)


@register_op("ai.onnx", "Optionaltype")
def OptionalType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Optionaltype."""
    return record_op("Optionaltype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Outofmemoryerror")
def OutOfMemoryError(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Outofmemoryerror."""
    return record_op("Outofmemoryerror", [x] + list(args), kwargs)


@register_op("ai.onnx", "Parameterdict")
def ParameterDict(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Parameterdict."""
    return record_op("Parameterdict", [x] + list(args), kwargs)


@register_op("ai.onnx", "Pyobjecttype")
def PyObjectType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Pyobjecttype."""
    return record_op("Pyobjecttype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Pytorchfilereader")
def PyTorchFileReader(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Pytorchfilereader."""
    return record_op("Pytorchfilereader", [x] + list(args), kwargs)


@register_op("ai.onnx", "Pytorchfilewriter")
def PyTorchFileWriter(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Pytorchfilewriter."""
    return record_op("Pytorchfilewriter", [x] + list(args), kwargs)


@register_op("ai.onnx", "Rreftype")
def RRefType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rreftype."""
    return record_op("Rreftype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sum")
def SUM(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sum."""
    return record_op("Sum", [x] + list(args), kwargs)


@register_op("ai.onnx", "Scriptclass")
def ScriptClass(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scriptclass."""
    return record_op("Scriptclass", [x] + list(args), kwargs)


@register_op("ai.onnx", "Scriptclassfunction")
def ScriptClassFunction(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scriptclassfunction."""
    return record_op("Scriptclassfunction", [x] + list(args), kwargs)


@register_op("ai.onnx", "Scriptdict")
def ScriptDict(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scriptdict."""
    return record_op("Scriptdict", [x] + list(args), kwargs)


@register_op("ai.onnx", "Scriptdictiterator")
def ScriptDictIterator(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scriptdictiterator."""
    return record_op("Scriptdictiterator", [x] + list(args), kwargs)


@register_op("ai.onnx", "Scriptdictkeyiterator")
def ScriptDictKeyIterator(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scriptdictkeyiterator."""
    return record_op("Scriptdictkeyiterator", [x] + list(args), kwargs)


@register_op("ai.onnx", "Scriptfunction")
def ScriptFunction(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scriptfunction."""
    return record_op("Scriptfunction", [x] + list(args), kwargs)


@register_op("ai.onnx", "Scriptlist")
def ScriptList(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scriptlist."""
    return record_op("Scriptlist", [x] + list(args), kwargs)


@register_op("ai.onnx", "Scriptlistiterator")
def ScriptListIterator(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scriptlistiterator."""
    return record_op("Scriptlistiterator", [x] + list(args), kwargs)


@register_op("ai.onnx", "Scriptmethod")
def ScriptMethod(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scriptmethod."""
    return record_op("Scriptmethod", [x] + list(args), kwargs)


@register_op("ai.onnx", "Scriptmodule")
def ScriptModule(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scriptmodule."""
    return record_op("Scriptmodule", [x] + list(args), kwargs)


@register_op("ai.onnx", "Scriptmoduleserializer")
def ScriptModuleSerializer(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scriptmoduleserializer."""
    return record_op("Scriptmoduleserializer", [x] + list(args), kwargs)


@register_op("ai.onnx", "Scriptobject")
def ScriptObject(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scriptobject."""
    return record_op("Scriptobject", [x] + list(args), kwargs)


@register_op("ai.onnx", "Scriptobjectproperty")
def ScriptObjectProperty(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scriptobjectproperty."""
    return record_op("Scriptobjectproperty", [x] + list(args), kwargs)


@register_op("ai.onnx", "Serializationstoragecontext")
def SerializationStorageContext(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Serializationstoragecontext."""
    return record_op("Serializationstoragecontext", [x] + list(args), kwargs)


@register_op("ai.onnx", "Size")
def Size(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Size."""
    return record_op("Size", [x] + list(args), kwargs)


@register_op("ai.onnx", "Staticmodule")
def StaticModule(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Staticmodule."""
    return record_op("Staticmodule", [x] + list(args), kwargs)


@register_op("ai.onnx", "Stream")
def Stream(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Stream."""
    return record_op("Stream", [x] + list(args), kwargs)


@register_op("ai.onnx", "Streamobjtype")
def StreamObjType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Streamobjtype."""
    return record_op("Streamobjtype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Stringtype")
def StringType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Stringtype."""
    return record_op("Stringtype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Symbooltype")
def SymBoolType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Symbooltype."""
    return record_op("Symbooltype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Syminttype")
def SymIntType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Syminttype."""
    return record_op("Syminttype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Tag")
def Tag(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Tag."""
    return record_op("Tag", [x] + list(args), kwargs)


@register_op("ai.onnx", "Tensortype")
def TensorType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Tensortype."""
    return record_op("Tensortype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Throughputbenchmark")
def ThroughputBenchmark(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Throughputbenchmark."""
    return record_op("Throughputbenchmark", [x] + list(args), kwargs)


@register_op("ai.onnx", "Tracingstate")
def TracingState(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Tracingstate."""
    return record_op("Tracingstate", [x] + list(args), kwargs)


@register_op("ai.onnx", "Tupletype")
def TupleType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Tupletype."""
    return record_op("Tupletype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Type")
def Type(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Type."""
    return record_op("Type", [x] + list(args), kwargs)


@register_op("ai.onnx", "Uniontype")
def UnionType(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Uniontype."""
    return record_op("Uniontype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Use")
def Use(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Use."""
    return record_op("Use", [x] + list(args), kwargs)


@register_op("ai.onnx", "Value")
def Value(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Value."""
    return record_op("Value", [x] + list(args), kwargs)


@register_op("ai.onnx", "Autocast_decrement_nesting")
def autocast_decrement_nesting(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Autocast_decrement_nesting."""
    return record_op("Autocast_decrement_nesting", [x] + list(args), kwargs)


@register_op("ai.onnx", "Autocast_increment_nesting")
def autocast_increment_nesting(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Autocast_increment_nesting."""
    return record_op("Autocast_increment_nesting", [x] + list(args), kwargs)


@register_op("ai.onnx", "Clear_autocast_cache")
def clear_autocast_cache(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Clear_autocast_cache."""
    return record_op("Clear_autocast_cache", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cpp_orderedmoduledict")
def cpp_OrderedModuleDict(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cpp_orderedmoduledict."""
    return record_op("Cpp_orderedmoduledict", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cpp_orderedtensordict")
def cpp_OrderedTensorDict(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cpp_orderedtensordict."""
    return record_op("Cpp_orderedtensordict", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cpp_nn_module")
def cpp_nn_Module(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cpp_nn_module."""
    return record_op("Cpp_nn_module", [x] + list(args), kwargs)


@register_op("ai.onnx", "Default_generator")
def default_generator(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Default_generator."""
    return record_op("Default_generator", [x] + list(args), kwargs)


@register_op("ai.onnx", "Device")
def device(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Device."""
    return record_op("Device", [x] + list(args), kwargs)


@register_op("ai.onnx", "Dtype")
def dtype(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Dtype."""
    return record_op("Dtype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Finfo")
def finfo(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Finfo."""
    return record_op("Finfo", [x] + list(args), kwargs)


@register_op("ai.onnx", "Fork")
def fork(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fork."""
    return record_op("Fork", [x] + list(args), kwargs)


@register_op("ai.onnx", "Get_autocast_cpu_dtype")
def get_autocast_cpu_dtype(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_autocast_cpu_dtype."""
    return record_op("Get_autocast_cpu_dtype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Get_autocast_dtype")
def get_autocast_dtype(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_autocast_dtype."""
    return record_op("Get_autocast_dtype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Get_autocast_gpu_dtype")
def get_autocast_gpu_dtype(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_autocast_gpu_dtype."""
    return record_op("Get_autocast_gpu_dtype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Get_autocast_ipu_dtype")
def get_autocast_ipu_dtype(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_autocast_ipu_dtype."""
    return record_op("Get_autocast_ipu_dtype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Get_autocast_xla_dtype")
def get_autocast_xla_dtype(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_autocast_xla_dtype."""
    return record_op("Get_autocast_xla_dtype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Get_default_dtype")
def get_default_dtype(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_default_dtype."""
    return record_op("Get_default_dtype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Get_num_interop_threads")
def get_num_interop_threads(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_num_interop_threads."""
    return record_op("Get_num_interop_threads", [x] + list(args), kwargs)


@register_op("ai.onnx", "Get_num_threads")
def get_num_threads(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_num_threads."""
    return record_op("Get_num_threads", [x] + list(args), kwargs)


@register_op("ai.onnx", "Has_lapack")
def has_lapack(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Has_lapack."""
    return record_op("Has_lapack", [x] + list(args), kwargs)


@register_op("ai.onnx", "Has_mkl")
def has_mkl(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Has_mkl."""
    return record_op("Has_mkl", [x] + list(args), kwargs)


@register_op("ai.onnx", "Has_openmp")
def has_openmp(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Has_openmp."""
    return record_op("Has_openmp", [x] + list(args), kwargs)


@register_op("ai.onnx", "Has_spectral")
def has_spectral(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Has_spectral."""
    return record_op("Has_spectral", [x] + list(args), kwargs)


@register_op("ai.onnx", "Iinfo")
def iinfo(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Iinfo."""
    return record_op("Iinfo", [x] + list(args), kwargs)


@register_op("ai.onnx", "Import_ir_module")
def import_ir_module(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Import_ir_module."""
    return record_op("Import_ir_module", [x] + list(args), kwargs)


@register_op("ai.onnx", "Import_ir_module_from_buffer")
def import_ir_module_from_buffer(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Import_ir_module_from_buffer."""
    return record_op("Import_ir_module_from_buffer", [x] + list(args), kwargs)


@register_op("ai.onnx", "Init_num_threads")
def init_num_threads(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Init_num_threads."""
    return record_op("Init_num_threads", [x] + list(args), kwargs)


@register_op("ai.onnx", "Is_anomaly_check_nan_enabled")
def is_anomaly_check_nan_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_anomaly_check_nan_enabled."""
    return record_op("Is_anomaly_check_nan_enabled", [x] + list(args), kwargs)


@register_op("ai.onnx", "Is_anomaly_enabled")
def is_anomaly_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_anomaly_enabled."""
    return record_op("Is_anomaly_enabled", [x] + list(args), kwargs)


@register_op("ai.onnx", "Is_autocast_cache_enabled")
def is_autocast_cache_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_autocast_cache_enabled."""
    return record_op("Is_autocast_cache_enabled", [x] + list(args), kwargs)


@register_op("ai.onnx", "Is_autocast_cpu_enabled")
def is_autocast_cpu_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_autocast_cpu_enabled."""
    return record_op("Is_autocast_cpu_enabled", [x] + list(args), kwargs)


@register_op("ai.onnx", "Is_autocast_enabled")
def is_autocast_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_autocast_enabled."""
    return record_op("Is_autocast_enabled", [x] + list(args), kwargs)


@register_op("ai.onnx", "Is_autocast_ipu_enabled")
def is_autocast_ipu_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_autocast_ipu_enabled."""
    return record_op("Is_autocast_ipu_enabled", [x] + list(args), kwargs)


@register_op("ai.onnx", "Is_autocast_xla_enabled")
def is_autocast_xla_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_autocast_xla_enabled."""
    return record_op("Is_autocast_xla_enabled", [x] + list(args), kwargs)


@register_op("ai.onnx", "Is_grad_enabled")
def is_grad_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_grad_enabled."""
    return record_op("Is_grad_enabled", [x] + list(args), kwargs)


@register_op("ai.onnx", "Is_inference_mode_enabled")
def is_inference_mode_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_inference_mode_enabled."""
    return record_op("Is_inference_mode_enabled", [x] + list(args), kwargs)


@register_op("ai.onnx", "Layout")
def layout(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Layout."""
    return record_op("Layout", [x] + list(args), kwargs)


@register_op("ai.onnx", "Memory_format")
def memory_format(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Memory_format."""
    return record_op("Memory_format", [x] + list(args), kwargs)


@register_op("ai.onnx", "Merge_type_from_type_comment")
def merge_type_from_type_comment(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Merge_type_from_type_comment."""
    return record_op("Merge_type_from_type_comment", [x] + list(args), kwargs)


@register_op("ai.onnx", "Parse_ir")
def parse_ir(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Parse_ir."""
    return record_op("Parse_ir", [x] + list(args), kwargs)


@register_op("ai.onnx", "Parse_schema")
def parse_schema(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Parse_schema."""
    return record_op("Parse_schema", [x] + list(args), kwargs)


@register_op("ai.onnx", "Parse_type_comment")
def parse_type_comment(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Parse_type_comment."""
    return record_op("Parse_type_comment", [x] + list(args), kwargs)


@register_op("ai.onnx", "Qscheme")
def qscheme(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Qscheme."""
    return record_op("Qscheme", [x] + list(args), kwargs)


@register_op("ai.onnx", "Read_vitals")
def read_vitals(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Read_vitals."""
    return record_op("Read_vitals", [x] + list(args), kwargs)


@register_op("ai.onnx", "Set_anomaly_enabled")
def set_anomaly_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_anomaly_enabled."""
    return record_op("Set_anomaly_enabled", [x] + list(args), kwargs)


@register_op("ai.onnx", "Set_autocast_cache_enabled")
def set_autocast_cache_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_autocast_cache_enabled."""
    return record_op("Set_autocast_cache_enabled", [x] + list(args), kwargs)


@register_op("ai.onnx", "Set_autocast_cpu_dtype")
def set_autocast_cpu_dtype(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_autocast_cpu_dtype."""
    return record_op("Set_autocast_cpu_dtype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Set_autocast_cpu_enabled")
def set_autocast_cpu_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_autocast_cpu_enabled."""
    return record_op("Set_autocast_cpu_enabled", [x] + list(args), kwargs)


@register_op("ai.onnx", "Set_autocast_dtype")
def set_autocast_dtype(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_autocast_dtype."""
    return record_op("Set_autocast_dtype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Set_autocast_enabled")
def set_autocast_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_autocast_enabled."""
    return record_op("Set_autocast_enabled", [x] + list(args), kwargs)


@register_op("ai.onnx", "Set_autocast_gpu_dtype")
def set_autocast_gpu_dtype(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_autocast_gpu_dtype."""
    return record_op("Set_autocast_gpu_dtype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Set_autocast_ipu_dtype")
def set_autocast_ipu_dtype(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_autocast_ipu_dtype."""
    return record_op("Set_autocast_ipu_dtype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Set_autocast_ipu_enabled")
def set_autocast_ipu_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_autocast_ipu_enabled."""
    return record_op("Set_autocast_ipu_enabled", [x] + list(args), kwargs)


@register_op("ai.onnx", "Set_autocast_xla_dtype")
def set_autocast_xla_dtype(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_autocast_xla_dtype."""
    return record_op("Set_autocast_xla_dtype", [x] + list(args), kwargs)


@register_op("ai.onnx", "Set_autocast_xla_enabled")
def set_autocast_xla_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_autocast_xla_enabled."""
    return record_op("Set_autocast_xla_enabled", [x] + list(args), kwargs)


@register_op("ai.onnx", "Set_flush_denormal")
def set_flush_denormal(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_flush_denormal."""
    return record_op("Set_flush_denormal", [x] + list(args), kwargs)


@register_op("ai.onnx", "Set_num_interop_threads")
def set_num_interop_threads(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_num_interop_threads."""
    return record_op("Set_num_interop_threads", [x] + list(args), kwargs)


@register_op("ai.onnx", "Set_num_threads")
def set_num_threads(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_num_threads."""
    return record_op("Set_num_threads", [x] + list(args), kwargs)


@register_op("ai.onnx", "Set_vital")
def set_vital(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Set_vital."""
    return record_op("Set_vital", [x] + list(args), kwargs)


@register_op("ai.onnx", "Unify_type_list")
def unify_type_list(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Unify_type_list."""
    return record_op("Unify_type_list", [x] + list(args), kwargs)


@register_op("ai.onnx", "Vitals_enabled")
def vitals_enabled(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Vitals_enabled."""
    return record_op("Vitals_enabled", [x] + list(args), kwargs)


@register_op("ai.onnx", "Wait")
def wait(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Wait."""
    return record_op("Wait", [x] + list(args), kwargs)


@register_op("ai.onnx", "E")
def e(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute E."""
    return record_op("E", [x] + list(args), kwargs)


@register_op("ai.onnx", "Pi")
def pi(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Pi."""
    return record_op("Pi", [x] + list(args), kwargs)


@register_op("ai.onnx", "Nan")
def nan(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Nan."""
    return record_op("Nan", [x] + list(args), kwargs)


@register_op("ai.onnx", "Inf")
def inf(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Inf."""
    return record_op("Inf", [x] + list(args), kwargs)


@register_op("ai.onnx", "Newaxis")
def newaxis(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Newaxis."""
    return record_op("Newaxis", [x] + list(args), kwargs)


@register_op("ai.onnx", "Abs_")
def abs_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Abs_."""
    return record_op("Abs_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Absolute")
def absolute(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Absolute."""
    return record_op("Absolute", [x] + list(args), kwargs)


@register_op("ai.onnx", "Acos_")
def acos_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Acos_."""
    return record_op("Acos_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Acosh_")
def acosh_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Acosh_."""
    return record_op("Acosh_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Adaptive_avg_pool1d")
def adaptive_avg_pool1d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Adaptive_avg_pool1d."""
    return record_op("Adaptive_avg_pool1d", [x] + list(args), kwargs)


@register_op("ai.onnx", "Adaptive_max_pool1d")
def adaptive_max_pool1d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Adaptive_max_pool1d."""
    return record_op("Adaptive_max_pool1d", [x] + list(args), kwargs)


@register_op("ai.onnx", "Addbmm")
def addbmm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Addbmm."""
    return record_op("Addbmm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Addcdiv")
def addcdiv(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Addcdiv."""
    return record_op("Addcdiv", [x] + list(args), kwargs)


@register_op("ai.onnx", "Addcmul")
def addcmul(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Addcmul."""
    return record_op("Addcmul", [x] + list(args), kwargs)


@register_op("ai.onnx", "Addmm")
def addmm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Addmm."""
    return record_op("Addmm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Addmv")
def addmv(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Addmv."""
    return record_op("Addmv", [x] + list(args), kwargs)


@register_op("ai.onnx", "Addmv_")
def addmv_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Addmv_."""
    return record_op("Addmv_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Addr")
def addr(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Addr."""
    return record_op("Addr", [x] + list(args), kwargs)


@register_op("ai.onnx", "Adjoint")
def adjoint(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Adjoint."""
    return record_op("Adjoint", [x] + list(args), kwargs)


@register_op("ai.onnx", "Affine_grid_generator")
def affine_grid_generator(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Affine_grid_generator."""
    return record_op("Affine_grid_generator", [x] + list(args), kwargs)


@register_op("ai.onnx", "Alias_copy")
def alias_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Alias_copy."""
    return record_op("Alias_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "Align_tensors")
def align_tensors(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Align_tensors."""
    return record_op("Align_tensors", [x] + list(args), kwargs)


@register_op("ai.onnx", "All")
def all(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute All."""
    return record_op("All", [x] + list(args), kwargs)


@register_op("ai.onnx", "Allclose")
def allclose(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Allclose."""
    return record_op("Allclose", [x] + list(args), kwargs)


@register_op("ai.onnx", "Alpha_dropout")
def alpha_dropout(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Alpha_dropout."""
    return record_op("Alpha_dropout", [x] + list(args), kwargs)


@register_op("ai.onnx", "Alpha_dropout_")
def alpha_dropout_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Alpha_dropout_."""
    return record_op("Alpha_dropout_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Amax")
def amax(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Amax."""
    return record_op("Amax", [x] + list(args), kwargs)


@register_op("ai.onnx", "Amin")
def amin(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Amin."""
    return record_op("Amin", [x] + list(args), kwargs)


@register_op("ai.onnx", "Aminmax")
def aminmax(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Aminmax."""
    return record_op("Aminmax", [x] + list(args), kwargs)


@register_op("ai.onnx", "Angle")
def angle(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Angle."""
    return record_op("Angle", [x] + list(args), kwargs)


@register_op("ai.onnx", "Any")
def any(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Any."""
    return record_op("Any", [x] + list(args), kwargs)


@register_op("ai.onnx", "Arange")
def arange(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arange."""
    return record_op("Arange", [x] + list(args), kwargs)


@register_op("ai.onnx", "Arccos")
def arccos(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arccos."""
    return record_op("Arccos", [x] + list(args), kwargs)


@register_op("ai.onnx", "Arccos_")
def arccos_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arccos_."""
    return record_op("Arccos_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Arccosh")
def arccosh(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arccosh."""
    return record_op("Arccosh", [x] + list(args), kwargs)


@register_op("ai.onnx", "Arccosh_")
def arccosh_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arccosh_."""
    return record_op("Arccosh_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Arcsin")
def arcsin(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arcsin."""
    return record_op("Arcsin", [x] + list(args), kwargs)


@register_op("ai.onnx", "Arcsin_")
def arcsin_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arcsin_."""
    return record_op("Arcsin_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Arcsinh")
def arcsinh(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arcsinh."""
    return record_op("Arcsinh", [x] + list(args), kwargs)


@register_op("ai.onnx", "Arcsinh_")
def arcsinh_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arcsinh_."""
    return record_op("Arcsinh_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Arctan")
def arctan(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arctan."""
    return record_op("Arctan", [x] + list(args), kwargs)


@register_op("ai.onnx", "Arctan2")
def arctan2(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arctan2."""
    return record_op("Arctan2", [x] + list(args), kwargs)


@register_op("ai.onnx", "Arctan_")
def arctan_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arctan_."""
    return record_op("Arctan_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Arctanh")
def arctanh(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arctanh."""
    return record_op("Arctanh", [x] + list(args), kwargs)


@register_op("ai.onnx", "Arctanh_")
def arctanh_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Arctanh_."""
    return record_op("Arctanh_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Argsort")
def argsort(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Argsort."""
    return record_op("Argsort", [x] + list(args), kwargs)


@register_op("ai.onnx", "Argwhere")
def argwhere(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Argwhere."""
    return record_op("Argwhere", [x] + list(args), kwargs)


@register_op("ai.onnx", "As_strided")
def as_strided(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute As_strided."""
    return record_op("As_strided", [x] + list(args), kwargs)


@register_op("ai.onnx", "As_strided_")
def as_strided_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute As_strided_."""
    return record_op("As_strided_", [x] + list(args), kwargs)


@register_op("ai.onnx", "As_strided_copy")
def as_strided_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute As_strided_copy."""
    return record_op("As_strided_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "As_strided_scatter")
def as_strided_scatter(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute As_strided_scatter."""
    return record_op("As_strided_scatter", [x] + list(args), kwargs)


@register_op("ai.onnx", "As_tensor")
def as_tensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute As_tensor."""
    return record_op("As_tensor", [x] + list(args), kwargs)


@register_op("ai.onnx", "Asarray")
def asarray(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Asarray."""
    return record_op("Asarray", [x] + list(args), kwargs)


@register_op("ai.onnx", "Asin_")
def asin_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Asin_."""
    return record_op("Asin_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Asinh_")
def asinh_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Asinh_."""
    return record_op("Asinh_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Atan2")
def atan2(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Atan2."""
    return record_op("Atan2", [x] + list(args), kwargs)


@register_op("ai.onnx", "Atan_")
def atan_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Atan_."""
    return record_op("Atan_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Atanh_")
def atanh_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Atanh_."""
    return record_op("Atanh_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Atleast_1d")
def atleast_1d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Atleast_1d."""
    return record_op("Atleast_1d", [x] + list(args), kwargs)


@register_op("ai.onnx", "Atleast_2d")
def atleast_2d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Atleast_2d."""
    return record_op("Atleast_2d", [x] + list(args), kwargs)


@register_op("ai.onnx", "Atleast_3d")
def atleast_3d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Atleast_3d."""
    return record_op("Atleast_3d", [x] + list(args), kwargs)


@register_op("ai.onnx", "Avg_pool1d")
def avg_pool1d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Avg_pool1d."""
    return record_op("Avg_pool1d", [x] + list(args), kwargs)


@register_op("ai.onnx", "Baddbmm")
def baddbmm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Baddbmm."""
    return record_op("Baddbmm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Bartlett_window")
def bartlett_window(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bartlett_window."""
    return record_op("Bartlett_window", [x] + list(args), kwargs)


@register_op("ai.onnx", "Batch_norm")
def batch_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Batch_norm."""
    return record_op("Batch_norm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Batch_norm_backward_elemt")
def batch_norm_backward_elemt(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Batch_norm_backward_elemt."""
    return record_op("Batch_norm_backward_elemt", [x] + list(args), kwargs)


@register_op("ai.onnx", "Batch_norm_backward_reduce")
def batch_norm_backward_reduce(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Batch_norm_backward_reduce."""
    return record_op("Batch_norm_backward_reduce", [x] + list(args), kwargs)


@register_op("ai.onnx", "Batch_norm_elemt")
def batch_norm_elemt(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Batch_norm_elemt."""
    return record_op("Batch_norm_elemt", [x] + list(args), kwargs)


@register_op("ai.onnx", "Batch_norm_gather_stats")
def batch_norm_gather_stats(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Batch_norm_gather_stats."""
    return record_op("Batch_norm_gather_stats", [x] + list(args), kwargs)


@register_op("ai.onnx", "Batch_norm_gather_stats_with_counts")
def batch_norm_gather_stats_with_counts(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Batch_norm_gather_stats_with_counts."""
    return record_op("Batch_norm_gather_stats_with_counts", [x] + list(args), kwargs)


@register_op("ai.onnx", "Batch_norm_stats")
def batch_norm_stats(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Batch_norm_stats."""
    return record_op("Batch_norm_stats", [x] + list(args), kwargs)


@register_op("ai.onnx", "Batch_norm_update_stats")
def batch_norm_update_stats(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Batch_norm_update_stats."""
    return record_op("Batch_norm_update_stats", [x] + list(args), kwargs)


@register_op("ai.onnx", "Bilinear")
def bilinear(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bilinear."""
    return record_op("Bilinear", [x] + list(args), kwargs)


@register_op("ai.onnx", "Binary_cross_entropy_with_logits")
def binary_cross_entropy_with_logits(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Binary_cross_entropy_with_logits."""
    return record_op("Binary_cross_entropy_with_logits", [x] + list(args), kwargs)


@register_op("ai.onnx", "Bincount")
def bincount(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bincount."""
    return record_op("Bincount", [x] + list(args), kwargs)


@register_op("ai.onnx", "Binomial")
def binomial(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Binomial."""
    return record_op("Binomial", [x] + list(args), kwargs)


@register_op("ai.onnx", "Bitwise_left_shift")
def bitwise_left_shift(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bitwise_left_shift."""
    return record_op("Bitwise_left_shift", [x] + list(args), kwargs)


@register_op("ai.onnx", "Bitwise_right_shift")
def bitwise_right_shift(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bitwise_right_shift."""
    return record_op("Bitwise_right_shift", [x] + list(args), kwargs)


@register_op("ai.onnx", "Block_diag")
def block_diag(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Block_diag."""
    return record_op("Block_diag", [x] + list(args), kwargs)


@register_op("ai.onnx", "Bmm")
def bmm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bmm."""
    return record_op("Bmm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Broadcast_tensors")
def broadcast_tensors(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Broadcast_tensors."""
    return record_op("Broadcast_tensors", [x] + list(args), kwargs)


@register_op("ai.onnx", "Broadcast_to")
def broadcast_to(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Broadcast_to."""
    return record_op("Broadcast_to", [x] + list(args), kwargs)


@register_op("ai.onnx", "Bucketize")
def bucketize(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bucketize."""
    return record_op("Bucketize", [x] + list(args), kwargs)


@register_op("ai.onnx", "Can_cast")
def can_cast(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Can_cast."""
    return record_op("Can_cast", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cartesian_prod")
def cartesian_prod(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cartesian_prod."""
    return record_op("Cartesian_prod", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cat")
def cat(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cat."""
    return record_op("Cat", [x] + list(args), kwargs)


@register_op("ai.onnx", "Ccol_indices_copy")
def ccol_indices_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Ccol_indices_copy."""
    return record_op("Ccol_indices_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cdist")
def cdist(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cdist."""
    return record_op("Cdist", [x] + list(args), kwargs)


@register_op("ai.onnx", "Ceil_")
def ceil_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Ceil_."""
    return record_op("Ceil_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Celu_")
def celu_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Celu_."""
    return record_op("Celu_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Chain_matmul")
def chain_matmul(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Chain_matmul."""
    return record_op("Chain_matmul", [x] + list(args), kwargs)


@register_op("ai.onnx", "Channel_shuffle")
def channel_shuffle(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Channel_shuffle."""
    return record_op("Channel_shuffle", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cholesky")
def cholesky(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cholesky."""
    return record_op("Cholesky", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cholesky_inverse")
def cholesky_inverse(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cholesky_inverse."""
    return record_op("Cholesky_inverse", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cholesky_solve")
def cholesky_solve(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cholesky_solve."""
    return record_op("Cholesky_solve", [x] + list(args), kwargs)


@register_op("ai.onnx", "Choose_qparams_optimized")
def choose_qparams_optimized(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Choose_qparams_optimized."""
    return record_op("Choose_qparams_optimized", [x] + list(args), kwargs)


@register_op("ai.onnx", "Clamp")
def clamp(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Clamp."""
    return record_op("Clamp", [x] + list(args), kwargs)


@register_op("ai.onnx", "Clamp_")
def clamp_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Clamp_."""
    return record_op("Clamp_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Clamp_max")
def clamp_max(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Clamp_max."""
    return record_op("Clamp_max", [x] + list(args), kwargs)


@register_op("ai.onnx", "Clamp_max_")
def clamp_max_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Clamp_max_."""
    return record_op("Clamp_max_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Clamp_min")
def clamp_min(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Clamp_min."""
    return record_op("Clamp_min", [x] + list(args), kwargs)


@register_op("ai.onnx", "Clamp_min_")
def clamp_min_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Clamp_min_."""
    return record_op("Clamp_min_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Clip_")
def clip_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Clip_."""
    return record_op("Clip_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Clone")
def clone(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Clone."""
    return record_op("Clone", [x] + list(args), kwargs)


@register_op("ai.onnx", "Col_indices_copy")
def col_indices_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Col_indices_copy."""
    return record_op("Col_indices_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "Column_stack")
def column_stack(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Column_stack."""
    return record_op("Column_stack", [x] + list(args), kwargs)


@register_op("ai.onnx", "Combinations")
def combinations(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Combinations."""
    return record_op("Combinations", [x] + list(args), kwargs)


@register_op("ai.onnx", "Complex")
def complex(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Complex."""
    return record_op("Complex", [x] + list(args), kwargs)


@register_op("ai.onnx", "Concatenate")
def concatenate(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Concatenate."""
    return record_op("Concatenate", [x] + list(args), kwargs)


@register_op("ai.onnx", "Conj")
def conj(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Conj."""
    return record_op("Conj", [x] + list(args), kwargs)


@register_op("ai.onnx", "Conj_physical")
def conj_physical(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Conj_physical."""
    return record_op("Conj_physical", [x] + list(args), kwargs)


@register_op("ai.onnx", "Conj_physical_")
def conj_physical_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Conj_physical_."""
    return record_op("Conj_physical_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Constant_pad_nd")
def constant_pad_nd(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Constant_pad_nd."""
    return record_op("Constant_pad_nd", [x] + list(args), kwargs)


@register_op("ai.onnx", "Conv1d")
def conv1d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Conv1d."""
    return record_op("Conv1d", [x] + list(args), kwargs)


@register_op("ai.onnx", "Conv2d")
def conv2d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Conv2d."""
    return record_op("Conv2d", [x] + list(args), kwargs)


@register_op("ai.onnx", "Conv3d")
def conv3d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Conv3d."""
    return record_op("Conv3d", [x] + list(args), kwargs)


@register_op("ai.onnx", "Conv_tbc")
def conv_tbc(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Conv_tbc."""
    return record_op("Conv_tbc", [x] + list(args), kwargs)


@register_op("ai.onnx", "Conv_transpose1d")
def conv_transpose1d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Conv_transpose1d."""
    return record_op("Conv_transpose1d", [x] + list(args), kwargs)


@register_op("ai.onnx", "Conv_transpose2d")
def conv_transpose2d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Conv_transpose2d."""
    return record_op("Conv_transpose2d", [x] + list(args), kwargs)


@register_op("ai.onnx", "Conv_transpose3d")
def conv_transpose3d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Conv_transpose3d."""
    return record_op("Conv_transpose3d", [x] + list(args), kwargs)


@register_op("ai.onnx", "Convolution")
def convolution(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Convolution."""
    return record_op("Convolution", [x] + list(args), kwargs)


@register_op("ai.onnx", "Copysign")
def copysign(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Copysign."""
    return record_op("Copysign", [x] + list(args), kwargs)


@register_op("ai.onnx", "Corrcoef")
def corrcoef(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Corrcoef."""
    return record_op("Corrcoef", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cos_")
def cos_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cos_."""
    return record_op("Cos_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cosh_")
def cosh_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cosh_."""
    return record_op("Cosh_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cosine_embedding_loss")
def cosine_embedding_loss(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cosine_embedding_loss."""
    return record_op("Cosine_embedding_loss", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cosine_similarity")
def cosine_similarity(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cosine_similarity."""
    return record_op("Cosine_similarity", [x] + list(args), kwargs)


@register_op("ai.onnx", "Count_nonzero")
def count_nonzero(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Count_nonzero."""
    return record_op("Count_nonzero", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cov")
def cov(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cov."""
    return record_op("Cov", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cross")
def cross(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cross."""
    return record_op("Cross", [x] + list(args), kwargs)


@register_op("ai.onnx", "Crow_indices_copy")
def crow_indices_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Crow_indices_copy."""
    return record_op("Crow_indices_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "Ctc_loss")
def ctc_loss(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Ctc_loss."""
    return record_op("Ctc_loss", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cudnn_affine_grid_generator")
def cudnn_affine_grid_generator(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cudnn_affine_grid_generator."""
    return record_op("Cudnn_affine_grid_generator", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cudnn_batch_norm")
def cudnn_batch_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cudnn_batch_norm."""
    return record_op("Cudnn_batch_norm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cudnn_convolution")
def cudnn_convolution(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cudnn_convolution."""
    return record_op("Cudnn_convolution", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cudnn_convolution_add_relu")
def cudnn_convolution_add_relu(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cudnn_convolution_add_relu."""
    return record_op("Cudnn_convolution_add_relu", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cudnn_convolution_relu")
def cudnn_convolution_relu(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cudnn_convolution_relu."""
    return record_op("Cudnn_convolution_relu", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cudnn_convolution_transpose")
def cudnn_convolution_transpose(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cudnn_convolution_transpose."""
    return record_op("Cudnn_convolution_transpose", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cudnn_grid_sampler")
def cudnn_grid_sampler(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cudnn_grid_sampler."""
    return record_op("Cudnn_grid_sampler", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cudnn_is_acceptable")
def cudnn_is_acceptable(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cudnn_is_acceptable."""
    return record_op("Cudnn_is_acceptable", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cummax")
def cummax(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cummax."""
    return record_op("Cummax", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cummin")
def cummin(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cummin."""
    return record_op("Cummin", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cumprod")
def cumprod(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cumprod."""
    return record_op("Cumprod", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cumulative_trapezoid")
def cumulative_trapezoid(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cumulative_trapezoid."""
    return record_op("Cumulative_trapezoid", [x] + list(args), kwargs)


@register_op("ai.onnx", "Deg2rad")
def deg2rad(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Deg2rad."""
    return record_op("Deg2rad", [x] + list(args), kwargs)


@register_op("ai.onnx", "Deg2rad_")
def deg2rad_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Deg2rad_."""
    return record_op("Deg2rad_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Dequantize")
def dequantize(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Dequantize."""
    return record_op("Dequantize", [x] + list(args), kwargs)


@register_op("ai.onnx", "Detach")
def detach(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Detach."""
    return record_op("Detach", [x] + list(args), kwargs)


@register_op("ai.onnx", "Detach_")
def detach_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Detach_."""
    return record_op("Detach_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Detach_copy")
def detach_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Detach_copy."""
    return record_op("Detach_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "Diag")
def diag(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Diag."""
    return record_op("Diag", [x] + list(args), kwargs)


@register_op("ai.onnx", "Diag_embed")
def diag_embed(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Diag_embed."""
    return record_op("Diag_embed", [x] + list(args), kwargs)


@register_op("ai.onnx", "Diagflat")
def diagflat(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Diagflat."""
    return record_op("Diagflat", [x] + list(args), kwargs)


@register_op("ai.onnx", "Diagonal")
def diagonal(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Diagonal."""
    return record_op("Diagonal", [x] + list(args), kwargs)


@register_op("ai.onnx", "Diagonal_copy")
def diagonal_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Diagonal_copy."""
    return record_op("Diagonal_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "Diagonal_scatter")
def diagonal_scatter(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Diagonal_scatter."""
    return record_op("Diagonal_scatter", [x] + list(args), kwargs)


@register_op("ai.onnx", "Diff")
def diff(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Diff."""
    return record_op("Diff", [x] + list(args), kwargs)


@register_op("ai.onnx", "Digamma")
def digamma(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Digamma."""
    return record_op("Digamma", [x] + list(args), kwargs)


@register_op("ai.onnx", "Dist")
def dist(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Dist."""
    return record_op("Dist", [x] + list(args), kwargs)


@register_op("ai.onnx", "Divide")
def divide(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Divide."""
    return record_op("Divide", [x] + list(args), kwargs)


@register_op("ai.onnx", "Dot")
def dot(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Dot."""
    return record_op("Dot", [x] + list(args), kwargs)


@register_op("ai.onnx", "Dropout_")
def dropout_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Dropout_."""
    return record_op("Dropout_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Dsmm")
def dsmm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Dsmm."""
    return record_op("Dsmm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Dsplit")
def dsplit(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Dsplit."""
    return record_op("Dsplit", [x] + list(args), kwargs)


@register_op("ai.onnx", "Dstack")
def dstack(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Dstack."""
    return record_op("Dstack", [x] + list(args), kwargs)


@register_op("ai.onnx", "Embedding")
def embedding(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Embedding."""
    return record_op("Embedding", [x] + list(args), kwargs)


@register_op("ai.onnx", "Embedding_bag")
def embedding_bag(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Embedding_bag."""
    return record_op("Embedding_bag", [x] + list(args), kwargs)


@register_op("ai.onnx", "Embedding_renorm_")
def embedding_renorm_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Embedding_renorm_."""
    return record_op("Embedding_renorm_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Empty")
def empty(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Empty."""
    return record_op("Empty", [x] + list(args), kwargs)


@register_op("ai.onnx", "Empty_like")
def empty_like(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Empty_like."""
    return record_op("Empty_like", [x] + list(args), kwargs)


@register_op("ai.onnx", "Empty_permuted")
def empty_permuted(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Empty_permuted."""
    return record_op("Empty_permuted", [x] + list(args), kwargs)


@register_op("ai.onnx", "Empty_quantized")
def empty_quantized(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Empty_quantized."""
    return record_op("Empty_quantized", [x] + list(args), kwargs)


@register_op("ai.onnx", "Empty_strided")
def empty_strided(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Empty_strided."""
    return record_op("Empty_strided", [x] + list(args), kwargs)


@register_op("ai.onnx", "Eq")
def eq(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Eq."""
    return record_op("Eq", [x] + list(args), kwargs)


@register_op("ai.onnx", "Erf_")
def erf_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Erf_."""
    return record_op("Erf_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Erfc")
def erfc(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Erfc."""
    return record_op("Erfc", [x] + list(args), kwargs)


@register_op("ai.onnx", "Erfc_")
def erfc_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Erfc_."""
    return record_op("Erfc_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Erfinv")
def erfinv(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Erfinv."""
    return record_op("Erfinv", [x] + list(args), kwargs)


@register_op("ai.onnx", "Exp2")
def exp2(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Exp2."""
    return record_op("Exp2", [x] + list(args), kwargs)


@register_op("ai.onnx", "Exp2_")
def exp2_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Exp2_."""
    return record_op("Exp2_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Exp_")
def exp_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Exp_."""
    return record_op("Exp_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Expand_copy")
def expand_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Expand_copy."""
    return record_op("Expand_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "Expm1")
def expm1(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Expm1."""
    return record_op("Expm1", [x] + list(args), kwargs)


@register_op("ai.onnx", "Expm1_")
def expm1_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Expm1_."""
    return record_op("Expm1_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Eye")
def eye(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Eye."""
    return record_op("Eye", [x] + list(args), kwargs)


@register_op("ai.onnx", "Fake_quantize_per_channel_affine")
def fake_quantize_per_channel_affine(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fake_quantize_per_channel_affine."""
    return record_op("Fake_quantize_per_channel_affine", [x] + list(args), kwargs)


@register_op("ai.onnx", "Fake_quantize_per_tensor_affine")
def fake_quantize_per_tensor_affine(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fake_quantize_per_tensor_affine."""
    return record_op("Fake_quantize_per_tensor_affine", [x] + list(args), kwargs)


@register_op("ai.onnx", "Fbgemm_linear_fp16_weight")
def fbgemm_linear_fp16_weight(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fbgemm_linear_fp16_weight."""
    return record_op("Fbgemm_linear_fp16_weight", [x] + list(args), kwargs)


@register_op("ai.onnx", "Fbgemm_linear_fp16_weight_fp32_activation")
def fbgemm_linear_fp16_weight_fp32_activation(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fbgemm_linear_fp16_weight_fp32_activation."""
    return record_op("Fbgemm_linear_fp16_weight_fp32_activation", [x] + list(args), kwargs)


@register_op("ai.onnx", "Fbgemm_linear_int8_weight")
def fbgemm_linear_int8_weight(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fbgemm_linear_int8_weight."""
    return record_op("Fbgemm_linear_int8_weight", [x] + list(args), kwargs)


@register_op("ai.onnx", "Fbgemm_linear_int8_weight_fp32_activation")
def fbgemm_linear_int8_weight_fp32_activation(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fbgemm_linear_int8_weight_fp32_activation."""
    return record_op("Fbgemm_linear_int8_weight_fp32_activation", [x] + list(args), kwargs)


@register_op("ai.onnx", "Fbgemm_linear_quantize_weight")
def fbgemm_linear_quantize_weight(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fbgemm_linear_quantize_weight."""
    return record_op("Fbgemm_linear_quantize_weight", [x] + list(args), kwargs)


@register_op("ai.onnx", "Fbgemm_pack_gemm_matrix_fp16")
def fbgemm_pack_gemm_matrix_fp16(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fbgemm_pack_gemm_matrix_fp16."""
    return record_op("Fbgemm_pack_gemm_matrix_fp16", [x] + list(args), kwargs)


@register_op("ai.onnx", "Fbgemm_pack_quantized_matrix")
def fbgemm_pack_quantized_matrix(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fbgemm_pack_quantized_matrix."""
    return record_op("Fbgemm_pack_quantized_matrix", [x] + list(args), kwargs)


@register_op("ai.onnx", "Feature_alpha_dropout")
def feature_alpha_dropout(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Feature_alpha_dropout."""
    return record_op("Feature_alpha_dropout", [x] + list(args), kwargs)


@register_op("ai.onnx", "Feature_alpha_dropout_")
def feature_alpha_dropout_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Feature_alpha_dropout_."""
    return record_op("Feature_alpha_dropout_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Feature_dropout")
def feature_dropout(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Feature_dropout."""
    return record_op("Feature_dropout", [x] + list(args), kwargs)


@register_op("ai.onnx", "Feature_dropout_")
def feature_dropout_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Feature_dropout_."""
    return record_op("Feature_dropout_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Fill")
def fill(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fill."""
    return record_op("Fill", [x] + list(args), kwargs)


@register_op("ai.onnx", "Fill_")
def fill_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fill_."""
    return record_op("Fill_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Fix")
def fix(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fix."""
    return record_op("Fix", [x] + list(args), kwargs)


@register_op("ai.onnx", "Fix_")
def fix_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fix_."""
    return record_op("Fix_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Flatten")
def flatten(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Flatten."""
    return record_op("Flatten", [x] + list(args), kwargs)


@register_op("ai.onnx", "Flip")
def flip(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Flip."""
    return record_op("Flip", [x] + list(args), kwargs)


@register_op("ai.onnx", "Fliplr")
def fliplr(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fliplr."""
    return record_op("Fliplr", [x] + list(args), kwargs)


@register_op("ai.onnx", "Flipud")
def flipud(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Flipud."""
    return record_op("Flipud", [x] + list(args), kwargs)


@register_op("ai.onnx", "Float_power")
def float_power(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Float_power."""
    return record_op("Float_power", [x] + list(args), kwargs)


@register_op("ai.onnx", "Floor_")
def floor_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Floor_."""
    return record_op("Floor_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Floor_divide")
def floor_divide(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Floor_divide."""
    return record_op("Floor_divide", [x] + list(args), kwargs)


@register_op("ai.onnx", "Fmax")
def fmax(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fmax."""
    return record_op("Fmax", [x] + list(args), kwargs)


@register_op("ai.onnx", "Fmin")
def fmin(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fmin."""
    return record_op("Fmin", [x] + list(args), kwargs)


@register_op("ai.onnx", "Fmod")
def fmod(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fmod."""
    return record_op("Fmod", [x] + list(args), kwargs)


@register_op("ai.onnx", "Frac")
def frac(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Frac."""
    return record_op("Frac", [x] + list(args), kwargs)


@register_op("ai.onnx", "Frac_")
def frac_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Frac_."""
    return record_op("Frac_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Frexp")
def frexp(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Frexp."""
    return record_op("Frexp", [x] + list(args), kwargs)


@register_op("ai.onnx", "Frobenius_norm")
def frobenius_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Frobenius_norm."""
    return record_op("Frobenius_norm", [x] + list(args), kwargs)


@register_op("ai.onnx", "From_file")
def from_file(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute From_file."""
    return record_op("From_file", [x] + list(args), kwargs)


@register_op("ai.onnx", "From_numpy")
def from_numpy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute From_numpy."""
    return record_op("From_numpy", [x] + list(args), kwargs)


@register_op("ai.onnx", "Frombuffer")
def frombuffer(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Frombuffer."""
    return record_op("Frombuffer", [x] + list(args), kwargs)


@register_op("ai.onnx", "Full")
def full(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Full."""
    return record_op("Full", [x] + list(args), kwargs)


@register_op("ai.onnx", "Full_like")
def full_like(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Full_like."""
    return record_op("Full_like", [x] + list(args), kwargs)


@register_op("ai.onnx", "Fused_moving_avg_obs_fake_quant")
def fused_moving_avg_obs_fake_quant(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Fused_moving_avg_obs_fake_quant."""
    return record_op("Fused_moving_avg_obs_fake_quant", [x] + list(args), kwargs)


@register_op("ai.onnx", "Gcd")
def gcd(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Gcd."""
    return record_op("Gcd", [x] + list(args), kwargs)


@register_op("ai.onnx", "Gcd_")
def gcd_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Gcd_."""
    return record_op("Gcd_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Ge")
def ge(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Ge."""
    return record_op("Ge", [x] + list(args), kwargs)


@register_op("ai.onnx", "Geqrf")
def geqrf(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Geqrf."""
    return record_op("Geqrf", [x] + list(args), kwargs)


@register_op("ai.onnx", "Ger")
def ger(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Ger."""
    return record_op("Ger", [x] + list(args), kwargs)


@register_op("ai.onnx", "Get_device")
def get_device(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Get_device."""
    return record_op("Get_device", [x] + list(args), kwargs)


@register_op("ai.onnx", "Gradient")
def gradient(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Gradient."""
    return record_op("Gradient", [x] + list(args), kwargs)


@register_op("ai.onnx", "Greater_equal")
def greater_equal(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Greater_equal."""
    return record_op("Greater_equal", [x] + list(args), kwargs)


@register_op("ai.onnx", "Grid_sampler")
def grid_sampler(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Grid_sampler."""
    return record_op("Grid_sampler", [x] + list(args), kwargs)


@register_op("ai.onnx", "Grid_sampler_2d")
def grid_sampler_2d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Grid_sampler_2d."""
    return record_op("Grid_sampler_2d", [x] + list(args), kwargs)


@register_op("ai.onnx", "Grid_sampler_3d")
def grid_sampler_3d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Grid_sampler_3d."""
    return record_op("Grid_sampler_3d", [x] + list(args), kwargs)


@register_op("ai.onnx", "Group_norm")
def group_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Group_norm."""
    return record_op("Group_norm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Gru_cell")
def gru_cell(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Gru_cell."""
    return record_op("Gru_cell", [x] + list(args), kwargs)


@register_op("ai.onnx", "Gt")
def gt(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Gt."""
    return record_op("Gt", [x] + list(args), kwargs)


@register_op("ai.onnx", "Hamming_window")
def hamming_window(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Hamming_window."""
    return record_op("Hamming_window", [x] + list(args), kwargs)


@register_op("ai.onnx", "Hann_window")
def hann_window(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Hann_window."""
    return record_op("Hann_window", [x] + list(args), kwargs)


@register_op("ai.onnx", "Hardshrink")
def hardshrink(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Hardshrink."""
    return record_op("Hardshrink", [x] + list(args), kwargs)


@register_op("ai.onnx", "Hash_tensor")
def hash_tensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Hash_tensor."""
    return record_op("Hash_tensor", [x] + list(args), kwargs)


@register_op("ai.onnx", "Heaviside")
def heaviside(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Heaviside."""
    return record_op("Heaviside", [x] + list(args), kwargs)


@register_op("ai.onnx", "Hinge_embedding_loss")
def hinge_embedding_loss(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Hinge_embedding_loss."""
    return record_op("Hinge_embedding_loss", [x] + list(args), kwargs)


@register_op("ai.onnx", "Histc")
def histc(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Histc."""
    return record_op("Histc", [x] + list(args), kwargs)


@register_op("ai.onnx", "Histogram")
def histogram(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Histogram."""
    return record_op("Histogram", [x] + list(args), kwargs)


@register_op("ai.onnx", "Histogramdd")
def histogramdd(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Histogramdd."""
    return record_op("Histogramdd", [x] + list(args), kwargs)


@register_op("ai.onnx", "Hsmm")
def hsmm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Hsmm."""
    return record_op("Hsmm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Hsplit")
def hsplit(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Hsplit."""
    return record_op("Hsplit", [x] + list(args), kwargs)


@register_op("ai.onnx", "Hspmm")
def hspmm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Hspmm."""
    return record_op("Hspmm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Hstack")
def hstack(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Hstack."""
    return record_op("Hstack", [x] + list(args), kwargs)


@register_op("ai.onnx", "Hypot")
def hypot(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Hypot."""
    return record_op("Hypot", [x] + list(args), kwargs)


@register_op("ai.onnx", "I0")
def i0(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute I0."""
    return record_op("I0", [x] + list(args), kwargs)


@register_op("ai.onnx", "I0_")
def i0_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute I0_."""
    return record_op("I0_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Igamma")
def igamma(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Igamma."""
    return record_op("Igamma", [x] + list(args), kwargs)


@register_op("ai.onnx", "Igammac")
def igammac(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Igammac."""
    return record_op("Igammac", [x] + list(args), kwargs)


@register_op("ai.onnx", "Imag")
def imag(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Imag."""
    return record_op("Imag", [x] + list(args), kwargs)


@register_op("ai.onnx", "Index_add")
def index_add(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Index_add."""
    return record_op("Index_add", [x] + list(args), kwargs)


@register_op("ai.onnx", "Index_copy")
def index_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Index_copy."""
    return record_op("Index_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "Index_fill")
def index_fill(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Index_fill."""
    return record_op("Index_fill", [x] + list(args), kwargs)


@register_op("ai.onnx", "Index_put")
def index_put(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Index_put."""
    return record_op("Index_put", [x] + list(args), kwargs)


@register_op("ai.onnx", "Index_put_")
def index_put_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Index_put_."""
    return record_op("Index_put_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Index_reduce")
def index_reduce(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Index_reduce."""
    return record_op("Index_reduce", [x] + list(args), kwargs)


@register_op("ai.onnx", "Index_select")
def index_select(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Index_select."""
    return record_op("Index_select", [x] + list(args), kwargs)


@register_op("ai.onnx", "Indices_copy")
def indices_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Indices_copy."""
    return record_op("Indices_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "Inner")
def inner(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Inner."""
    return record_op("Inner", [x] + list(args), kwargs)


@register_op("ai.onnx", "Instance_norm")
def instance_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Instance_norm."""
    return record_op("Instance_norm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Int_repr")
def int_repr(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Int_repr."""
    return record_op("Int_repr", [x] + list(args), kwargs)


@register_op("ai.onnx", "Inverse")
def inverse(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Inverse."""
    return record_op("Inverse", [x] + list(args), kwargs)


@register_op("ai.onnx", "Is_complex")
def is_complex(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_complex."""
    return record_op("Is_complex", [x] + list(args), kwargs)


@register_op("ai.onnx", "Is_conj")
def is_conj(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_conj."""
    return record_op("Is_conj", [x] + list(args), kwargs)


@register_op("ai.onnx", "Is_distributed")
def is_distributed(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_distributed."""
    return record_op("Is_distributed", [x] + list(args), kwargs)


@register_op("ai.onnx", "Is_floating_point")
def is_floating_point(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_floating_point."""
    return record_op("Is_floating_point", [x] + list(args), kwargs)


@register_op("ai.onnx", "Is_inference")
def is_inference(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_inference."""
    return record_op("Is_inference", [x] + list(args), kwargs)


@register_op("ai.onnx", "Is_neg")
def is_neg(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_neg."""
    return record_op("Is_neg", [x] + list(args), kwargs)


@register_op("ai.onnx", "Is_nonzero")
def is_nonzero(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_nonzero."""
    return record_op("Is_nonzero", [x] + list(args), kwargs)


@register_op("ai.onnx", "Is_same_size")
def is_same_size(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_same_size."""
    return record_op("Is_same_size", [x] + list(args), kwargs)


@register_op("ai.onnx", "Is_signed")
def is_signed(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_signed."""
    return record_op("Is_signed", [x] + list(args), kwargs)


@register_op("ai.onnx", "Is_vulkan_available")
def is_vulkan_available(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Is_vulkan_available."""
    return record_op("Is_vulkan_available", [x] + list(args), kwargs)


@register_op("ai.onnx", "Isclose")
def isclose(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Isclose."""
    return record_op("Isclose", [x] + list(args), kwargs)


@register_op("ai.onnx", "Isfinite")
def isfinite(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Isfinite."""
    return record_op("Isfinite", [x] + list(args), kwargs)


@register_op("ai.onnx", "Isin")
def isin(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Isin."""
    return record_op("Isin", [x] + list(args), kwargs)


@register_op("ai.onnx", "Isneginf")
def isneginf(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Isneginf."""
    return record_op("Isneginf", [x] + list(args), kwargs)


@register_op("ai.onnx", "Isposinf")
def isposinf(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Isposinf."""
    return record_op("Isposinf", [x] + list(args), kwargs)


@register_op("ai.onnx", "Isreal")
def isreal(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Isreal."""
    return record_op("Isreal", [x] + list(args), kwargs)


@register_op("ai.onnx", "Istft")
def istft(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Istft."""
    return record_op("Istft", [x] + list(args), kwargs)


@register_op("ai.onnx", "Kaiser_window")
def kaiser_window(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Kaiser_window."""
    return record_op("Kaiser_window", [x] + list(args), kwargs)


@register_op("ai.onnx", "Kl_div")
def kl_div(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Kl_div."""
    return record_op("Kl_div", [x] + list(args), kwargs)


@register_op("ai.onnx", "Kron")
def kron(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Kron."""
    return record_op("Kron", [x] + list(args), kwargs)


@register_op("ai.onnx", "Kthvalue")
def kthvalue(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Kthvalue."""
    return record_op("Kthvalue", [x] + list(args), kwargs)


@register_op("ai.onnx", "Layer_norm")
def layer_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Layer_norm."""
    return record_op("Layer_norm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Lcm")
def lcm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Lcm."""
    return record_op("Lcm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Lcm_")
def lcm_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Lcm_."""
    return record_op("Lcm_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Ldexp")
def ldexp(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Ldexp."""
    return record_op("Ldexp", [x] + list(args), kwargs)


@register_op("ai.onnx", "Ldexp_")
def ldexp_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Ldexp_."""
    return record_op("Ldexp_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Le")
def le(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Le."""
    return record_op("Le", [x] + list(args), kwargs)


@register_op("ai.onnx", "Lerp")
def lerp(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Lerp."""
    return record_op("Lerp", [x] + list(args), kwargs)


@register_op("ai.onnx", "Less_equal")
def less_equal(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Less_equal."""
    return record_op("Less_equal", [x] + list(args), kwargs)


@register_op("ai.onnx", "Lgamma")
def lgamma(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Lgamma."""
    return record_op("Lgamma", [x] + list(args), kwargs)


@register_op("ai.onnx", "Linspace")
def linspace(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Linspace."""
    return record_op("Linspace", [x] + list(args), kwargs)


@register_op("ai.onnx", "Log")
def log(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Log."""
    return record_op("Log", [x] + list(args), kwargs)


@register_op("ai.onnx", "Log10")
def log10(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Log10."""
    return record_op("Log10", [x] + list(args), kwargs)


@register_op("ai.onnx", "Log10_")
def log10_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Log10_."""
    return record_op("Log10_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Log1p")
def log1p(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Log1p."""
    return record_op("Log1p", [x] + list(args), kwargs)


@register_op("ai.onnx", "Log1p_")
def log1p_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Log1p_."""
    return record_op("Log1p_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Log2")
def log2(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Log2."""
    return record_op("Log2", [x] + list(args), kwargs)


@register_op("ai.onnx", "Log2_")
def log2_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Log2_."""
    return record_op("Log2_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Log_")
def log_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Log_."""
    return record_op("Log_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Logaddexp")
def logaddexp(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Logaddexp."""
    return record_op("Logaddexp", [x] + list(args), kwargs)


@register_op("ai.onnx", "Logaddexp2")
def logaddexp2(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Logaddexp2."""
    return record_op("Logaddexp2", [x] + list(args), kwargs)


@register_op("ai.onnx", "Logcumsumexp")
def logcumsumexp(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Logcumsumexp."""
    return record_op("Logcumsumexp", [x] + list(args), kwargs)


@register_op("ai.onnx", "Logdet")
def logdet(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Logdet."""
    return record_op("Logdet", [x] + list(args), kwargs)


@register_op("ai.onnx", "Logical_and")
def logical_and(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Logical_and."""
    return record_op("Logical_and", [x] + list(args), kwargs)


@register_op("ai.onnx", "Logical_not")
def logical_not(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Logical_not."""
    return record_op("Logical_not", [x] + list(args), kwargs)


@register_op("ai.onnx", "Logical_or")
def logical_or(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Logical_or."""
    return record_op("Logical_or", [x] + list(args), kwargs)


@register_op("ai.onnx", "Logical_xor")
def logical_xor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Logical_xor."""
    return record_op("Logical_xor", [x] + list(args), kwargs)


@register_op("ai.onnx", "Logit")
def logit(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Logit."""
    return record_op("Logit", [x] + list(args), kwargs)


@register_op("ai.onnx", "Logit_")
def logit_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Logit_."""
    return record_op("Logit_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Logspace")
def logspace(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Logspace."""
    return record_op("Logspace", [x] + list(args), kwargs)


@register_op("ai.onnx", "Logsumexp")
def logsumexp(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Logsumexp."""
    return record_op("Logsumexp", [x] + list(args), kwargs)


@register_op("ai.onnx", "Lstm_cell")
def lstm_cell(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Lstm_cell."""
    return record_op("Lstm_cell", [x] + list(args), kwargs)


@register_op("ai.onnx", "Lt")
def lt(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Lt."""
    return record_op("Lt", [x] + list(args), kwargs)


@register_op("ai.onnx", "Lu_solve")
def lu_solve(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Lu_solve."""
    return record_op("Lu_solve", [x] + list(args), kwargs)


@register_op("ai.onnx", "Lu_unpack")
def lu_unpack(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Lu_unpack."""
    return record_op("Lu_unpack", [x] + list(args), kwargs)


@register_op("ai.onnx", "Margin_ranking_loss")
def margin_ranking_loss(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Margin_ranking_loss."""
    return record_op("Margin_ranking_loss", [x] + list(args), kwargs)


@register_op("ai.onnx", "Masked_fill")
def masked_fill(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Masked_fill."""
    return record_op("Masked_fill", [x] + list(args), kwargs)


@register_op("ai.onnx", "Masked_scatter")
def masked_scatter(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Masked_scatter."""
    return record_op("Masked_scatter", [x] + list(args), kwargs)


@register_op("ai.onnx", "Masked_select")
def masked_select(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Masked_select."""
    return record_op("Masked_select", [x] + list(args), kwargs)


@register_op("ai.onnx", "Matrix_exp")
def matrix_exp(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Matrix_exp."""
    return record_op("Matrix_exp", [x] + list(args), kwargs)


@register_op("ai.onnx", "Matrix_power")
def matrix_power(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Matrix_power."""
    return record_op("Matrix_power", [x] + list(args), kwargs)


@register_op("ai.onnx", "Max_pool1d")
def max_pool1d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Max_pool1d."""
    return record_op("Max_pool1d", [x] + list(args), kwargs)


@register_op("ai.onnx", "Max_pool1d_with_indices")
def max_pool1d_with_indices(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Max_pool1d_with_indices."""
    return record_op("Max_pool1d_with_indices", [x] + list(args), kwargs)


@register_op("ai.onnx", "Max_pool2d")
def max_pool2d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Max_pool2d."""
    return record_op("Max_pool2d", [x] + list(args), kwargs)


@register_op("ai.onnx", "Max_pool3d")
def max_pool3d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Max_pool3d."""
    return record_op("Max_pool3d", [x] + list(args), kwargs)


@register_op("ai.onnx", "Maximum")
def maximum(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Maximum."""
    return record_op("Maximum", [x] + list(args), kwargs)


@register_op("ai.onnx", "Median")
def median(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Median."""
    return record_op("Median", [x] + list(args), kwargs)


@register_op("ai.onnx", "Meshgrid")
def meshgrid(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Meshgrid."""
    return record_op("Meshgrid", [x] + list(args), kwargs)


@register_op("ai.onnx", "Minimum")
def minimum(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Minimum."""
    return record_op("Minimum", [x] + list(args), kwargs)


@register_op("ai.onnx", "Miopen_batch_norm")
def miopen_batch_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Miopen_batch_norm."""
    return record_op("Miopen_batch_norm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Miopen_convolution")
def miopen_convolution(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Miopen_convolution."""
    return record_op("Miopen_convolution", [x] + list(args), kwargs)


@register_op("ai.onnx", "Miopen_convolution_add_relu")
def miopen_convolution_add_relu(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Miopen_convolution_add_relu."""
    return record_op("Miopen_convolution_add_relu", [x] + list(args), kwargs)


@register_op("ai.onnx", "Miopen_convolution_relu")
def miopen_convolution_relu(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Miopen_convolution_relu."""
    return record_op("Miopen_convolution_relu", [x] + list(args), kwargs)


@register_op("ai.onnx", "Miopen_convolution_transpose")
def miopen_convolution_transpose(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Miopen_convolution_transpose."""
    return record_op("Miopen_convolution_transpose", [x] + list(args), kwargs)


@register_op("ai.onnx", "Miopen_ctc_loss")
def miopen_ctc_loss(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Miopen_ctc_loss."""
    return record_op("Miopen_ctc_loss", [x] + list(args), kwargs)


@register_op("ai.onnx", "Miopen_depthwise_convolution")
def miopen_depthwise_convolution(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Miopen_depthwise_convolution."""
    return record_op("Miopen_depthwise_convolution", [x] + list(args), kwargs)


@register_op("ai.onnx", "Miopen_rnn")
def miopen_rnn(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Miopen_rnn."""
    return record_op("Miopen_rnn", [x] + list(args), kwargs)


@register_op("ai.onnx", "Mkldnn_adaptive_avg_pool2d")
def mkldnn_adaptive_avg_pool2d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Mkldnn_adaptive_avg_pool2d."""
    return record_op("Mkldnn_adaptive_avg_pool2d", [x] + list(args), kwargs)


@register_op("ai.onnx", "Mkldnn_convolution")
def mkldnn_convolution(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Mkldnn_convolution."""
    return record_op("Mkldnn_convolution", [x] + list(args), kwargs)


@register_op("ai.onnx", "Mkldnn_linear_backward_weights")
def mkldnn_linear_backward_weights(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Mkldnn_linear_backward_weights."""
    return record_op("Mkldnn_linear_backward_weights", [x] + list(args), kwargs)


@register_op("ai.onnx", "Mkldnn_max_pool2d")
def mkldnn_max_pool2d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Mkldnn_max_pool2d."""
    return record_op("Mkldnn_max_pool2d", [x] + list(args), kwargs)


@register_op("ai.onnx", "Mkldnn_max_pool3d")
def mkldnn_max_pool3d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Mkldnn_max_pool3d."""
    return record_op("Mkldnn_max_pool3d", [x] + list(args), kwargs)


@register_op("ai.onnx", "Mkldnn_rnn_layer")
def mkldnn_rnn_layer(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Mkldnn_rnn_layer."""
    return record_op("Mkldnn_rnn_layer", [x] + list(args), kwargs)


@register_op("ai.onnx", "Mm")
def mm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Mm."""
    return record_op("Mm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Mode")
def mode(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Mode."""
    return record_op("Mode", [x] + list(args), kwargs)


@register_op("ai.onnx", "Moveaxis")
def moveaxis(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Moveaxis."""
    return record_op("Moveaxis", [x] + list(args), kwargs)


@register_op("ai.onnx", "Movedim")
def movedim(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Movedim."""
    return record_op("Movedim", [x] + list(args), kwargs)


@register_op("ai.onnx", "Msort")
def msort(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Msort."""
    return record_op("Msort", [x] + list(args), kwargs)


@register_op("ai.onnx", "Multiply")
def multiply(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Multiply."""
    return record_op("Multiply", [x] + list(args), kwargs)


@register_op("ai.onnx", "Mv")
def mv(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Mv."""
    return record_op("Mv", [x] + list(args), kwargs)


@register_op("ai.onnx", "Mvlgamma")
def mvlgamma(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Mvlgamma."""
    return record_op("Mvlgamma", [x] + list(args), kwargs)


@register_op("ai.onnx", "Nan_to_num")
def nan_to_num(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Nan_to_num."""
    return record_op("Nan_to_num", [x] + list(args), kwargs)


@register_op("ai.onnx", "Nan_to_num_")
def nan_to_num_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Nan_to_num_."""
    return record_op("Nan_to_num_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Nanmean")
def nanmean(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Nanmean."""
    return record_op("Nanmean", [x] + list(args), kwargs)


@register_op("ai.onnx", "Nanmedian")
def nanmedian(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Nanmedian."""
    return record_op("Nanmedian", [x] + list(args), kwargs)


@register_op("ai.onnx", "Nanquantile")
def nanquantile(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Nanquantile."""
    return record_op("Nanquantile", [x] + list(args), kwargs)


@register_op("ai.onnx", "Nansum")
def nansum(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Nansum."""
    return record_op("Nansum", [x] + list(args), kwargs)


@register_op("ai.onnx", "Narrow")
def narrow(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Narrow."""
    return record_op("Narrow", [x] + list(args), kwargs)


@register_op("ai.onnx", "Narrow_copy")
def narrow_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Narrow_copy."""
    return record_op("Narrow_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "Native_batch_norm")
def native_batch_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Native_batch_norm."""
    return record_op("Native_batch_norm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Native_channel_shuffle")
def native_channel_shuffle(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Native_channel_shuffle."""
    return record_op("Native_channel_shuffle", [x] + list(args), kwargs)


@register_op("ai.onnx", "Native_dropout")
def native_dropout(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Native_dropout."""
    return record_op("Native_dropout", [x] + list(args), kwargs)


@register_op("ai.onnx", "Native_group_norm")
def native_group_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Native_group_norm."""
    return record_op("Native_group_norm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Native_layer_norm")
def native_layer_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Native_layer_norm."""
    return record_op("Native_layer_norm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Native_norm")
def native_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Native_norm."""
    return record_op("Native_norm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Ne")
def ne(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Ne."""
    return record_op("Ne", [x] + list(args), kwargs)


@register_op("ai.onnx", "Neg_")
def neg_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Neg_."""
    return record_op("Neg_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Negative")
def negative(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Negative."""
    return record_op("Negative", [x] + list(args), kwargs)


@register_op("ai.onnx", "Negative_")
def negative_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Negative_."""
    return record_op("Negative_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Nextafter")
def nextafter(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Nextafter."""
    return record_op("Nextafter", [x] + list(args), kwargs)


@register_op("ai.onnx", "Nonzero")
def nonzero(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Nonzero."""
    return record_op("Nonzero", [x] + list(args), kwargs)


@register_op("ai.onnx", "Nonzero_static")
def nonzero_static(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Nonzero_static."""
    return record_op("Nonzero_static", [x] + list(args), kwargs)


@register_op("ai.onnx", "Norm")
def norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Norm."""
    return record_op("Norm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Norm_except_dim")
def norm_except_dim(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Norm_except_dim."""
    return record_op("Norm_except_dim", [x] + list(args), kwargs)


@register_op("ai.onnx", "Normal")
def normal(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Normal."""
    return record_op("Normal", [x] + list(args), kwargs)


@register_op("ai.onnx", "Not_equal")
def not_equal(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Not_equal."""
    return record_op("Not_equal", [x] + list(args), kwargs)


@register_op("ai.onnx", "Nuclear_norm")
def nuclear_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Nuclear_norm."""
    return record_op("Nuclear_norm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Numel")
def numel(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Numel."""
    return record_op("Numel", [x] + list(args), kwargs)


@register_op("ai.onnx", "Ones_like")
def ones_like(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Ones_like."""
    return record_op("Ones_like", [x] + list(args), kwargs)


@register_op("ai.onnx", "Orgqr")
def orgqr(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Orgqr."""
    return record_op("Orgqr", [x] + list(args), kwargs)


@register_op("ai.onnx", "Ormqr")
def ormqr(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Ormqr."""
    return record_op("Ormqr", [x] + list(args), kwargs)


@register_op("ai.onnx", "Outer")
def outer(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Outer."""
    return record_op("Outer", [x] + list(args), kwargs)


@register_op("ai.onnx", "Pairwise_distance")
def pairwise_distance(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Pairwise_distance."""
    return record_op("Pairwise_distance", [x] + list(args), kwargs)


@register_op("ai.onnx", "Pdist")
def pdist(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Pdist."""
    return record_op("Pdist", [x] + list(args), kwargs)


@register_op("ai.onnx", "Permute")
def permute(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Permute."""
    return record_op("Permute", [x] + list(args), kwargs)


@register_op("ai.onnx", "Permute_copy")
def permute_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Permute_copy."""
    return record_op("Permute_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "Pinverse")
def pinverse(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Pinverse."""
    return record_op("Pinverse", [x] + list(args), kwargs)


@register_op("ai.onnx", "Pixel_shuffle")
def pixel_shuffle(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Pixel_shuffle."""
    return record_op("Pixel_shuffle", [x] + list(args), kwargs)


@register_op("ai.onnx", "Pixel_unshuffle")
def pixel_unshuffle(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Pixel_unshuffle."""
    return record_op("Pixel_unshuffle", [x] + list(args), kwargs)


@register_op("ai.onnx", "Poisson")
def poisson(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Poisson."""
    return record_op("Poisson", [x] + list(args), kwargs)


@register_op("ai.onnx", "Poisson_nll_loss")
def poisson_nll_loss(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Poisson_nll_loss."""
    return record_op("Poisson_nll_loss", [x] + list(args), kwargs)


@register_op("ai.onnx", "Polar")
def polar(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Polar."""
    return record_op("Polar", [x] + list(args), kwargs)


@register_op("ai.onnx", "Polygamma")
def polygamma(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Polygamma."""
    return record_op("Polygamma", [x] + list(args), kwargs)


@register_op("ai.onnx", "Positive")
def positive(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Positive."""
    return record_op("Positive", [x] + list(args), kwargs)


@register_op("ai.onnx", "Prod")
def prod(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Prod."""
    return record_op("Prod", [x] + list(args), kwargs)


@register_op("ai.onnx", "Promote_types")
def promote_types(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Promote_types."""
    return record_op("Promote_types", [x] + list(args), kwargs)


@register_op("ai.onnx", "Put")
def put(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Put."""
    return record_op("Put", [x] + list(args), kwargs)


@register_op("ai.onnx", "Q_per_channel_axis")
def q_per_channel_axis(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Q_per_channel_axis."""
    return record_op("Q_per_channel_axis", [x] + list(args), kwargs)


@register_op("ai.onnx", "Q_per_channel_scales")
def q_per_channel_scales(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Q_per_channel_scales."""
    return record_op("Q_per_channel_scales", [x] + list(args), kwargs)


@register_op("ai.onnx", "Q_per_channel_zero_points")
def q_per_channel_zero_points(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Q_per_channel_zero_points."""
    return record_op("Q_per_channel_zero_points", [x] + list(args), kwargs)


@register_op("ai.onnx", "Q_scale")
def q_scale(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Q_scale."""
    return record_op("Q_scale", [x] + list(args), kwargs)


@register_op("ai.onnx", "Q_zero_point")
def q_zero_point(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Q_zero_point."""
    return record_op("Q_zero_point", [x] + list(args), kwargs)


@register_op("ai.onnx", "Qr")
def qr(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Qr."""
    return record_op("Qr", [x] + list(args), kwargs)


@register_op("ai.onnx", "Quantile")
def quantile(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quantile."""
    return record_op("Quantile", [x] + list(args), kwargs)


@register_op("ai.onnx", "Quantize_per_channel")
def quantize_per_channel(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quantize_per_channel."""
    return record_op("Quantize_per_channel", [x] + list(args), kwargs)


@register_op("ai.onnx", "Quantize_per_tensor")
def quantize_per_tensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quantize_per_tensor."""
    return record_op("Quantize_per_tensor", [x] + list(args), kwargs)


@register_op("ai.onnx", "Quantize_per_tensor_dynamic")
def quantize_per_tensor_dynamic(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quantize_per_tensor_dynamic."""
    return record_op("Quantize_per_tensor_dynamic", [x] + list(args), kwargs)


@register_op("ai.onnx", "Quantized_batch_norm")
def quantized_batch_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quantized_batch_norm."""
    return record_op("Quantized_batch_norm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Quantized_gru_cell")
def quantized_gru_cell(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quantized_gru_cell."""
    return record_op("Quantized_gru_cell", [x] + list(args), kwargs)


@register_op("ai.onnx", "Quantized_lstm_cell")
def quantized_lstm_cell(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quantized_lstm_cell."""
    return record_op("Quantized_lstm_cell", [x] + list(args), kwargs)


@register_op("ai.onnx", "Quantized_max_pool1d")
def quantized_max_pool1d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quantized_max_pool1d."""
    return record_op("Quantized_max_pool1d", [x] + list(args), kwargs)


@register_op("ai.onnx", "Quantized_max_pool2d")
def quantized_max_pool2d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quantized_max_pool2d."""
    return record_op("Quantized_max_pool2d", [x] + list(args), kwargs)


@register_op("ai.onnx", "Quantized_max_pool3d")
def quantized_max_pool3d(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quantized_max_pool3d."""
    return record_op("Quantized_max_pool3d", [x] + list(args), kwargs)


@register_op("ai.onnx", "Quantized_rnn_relu_cell")
def quantized_rnn_relu_cell(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quantized_rnn_relu_cell."""
    return record_op("Quantized_rnn_relu_cell", [x] + list(args), kwargs)


@register_op("ai.onnx", "Quantized_rnn_tanh_cell")
def quantized_rnn_tanh_cell(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quantized_rnn_tanh_cell."""
    return record_op("Quantized_rnn_tanh_cell", [x] + list(args), kwargs)


@register_op("ai.onnx", "Rad2deg")
def rad2deg(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rad2deg."""
    return record_op("Rad2deg", [x] + list(args), kwargs)


@register_op("ai.onnx", "Rad2deg_")
def rad2deg_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rad2deg_."""
    return record_op("Rad2deg_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Rand_like")
def rand_like(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rand_like."""
    return record_op("Rand_like", [x] + list(args), kwargs)


@register_op("ai.onnx", "Randint")
def randint(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Randint."""
    return record_op("Randint", [x] + list(args), kwargs)


@register_op("ai.onnx", "Randint_like")
def randint_like(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Randint_like."""
    return record_op("Randint_like", [x] + list(args), kwargs)


@register_op("ai.onnx", "Randn_like")
def randn_like(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Randn_like."""
    return record_op("Randn_like", [x] + list(args), kwargs)


@register_op("ai.onnx", "Randperm")
def randperm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Randperm."""
    return record_op("Randperm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Range")
def range(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Range."""
    return record_op("Range", [x] + list(args), kwargs)


@register_op("ai.onnx", "Ravel")
def ravel(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Ravel."""
    return record_op("Ravel", [x] + list(args), kwargs)


@register_op("ai.onnx", "Real")
def real(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Real."""
    return record_op("Real", [x] + list(args), kwargs)


@register_op("ai.onnx", "Reciprocal_")
def reciprocal_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Reciprocal_."""
    return record_op("Reciprocal_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Relu_")
def relu_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Relu_."""
    return record_op("Relu_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Remainder")
def remainder(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Remainder."""
    return record_op("Remainder", [x] + list(args), kwargs)


@register_op("ai.onnx", "Renorm")
def renorm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Renorm."""
    return record_op("Renorm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Repeat_interleave")
def repeat_interleave(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Repeat_interleave."""
    return record_op("Repeat_interleave", [x] + list(args), kwargs)


@register_op("ai.onnx", "Resize_as_")
def resize_as_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Resize_as_."""
    return record_op("Resize_as_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Resize_as_sparse_")
def resize_as_sparse_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Resize_as_sparse_."""
    return record_op("Resize_as_sparse_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Resolve_conj")
def resolve_conj(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Resolve_conj."""
    return record_op("Resolve_conj", [x] + list(args), kwargs)


@register_op("ai.onnx", "Resolve_neg")
def resolve_neg(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Resolve_neg."""
    return record_op("Resolve_neg", [x] + list(args), kwargs)


@register_op("ai.onnx", "Result_type")
def result_type(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Result_type."""
    return record_op("Result_type", [x] + list(args), kwargs)


@register_op("ai.onnx", "Rms_norm")
def rms_norm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rms_norm."""
    return record_op("Rms_norm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Rnn_relu")
def rnn_relu(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rnn_relu."""
    return record_op("Rnn_relu", [x] + list(args), kwargs)


@register_op("ai.onnx", "Rnn_relu_cell")
def rnn_relu_cell(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rnn_relu_cell."""
    return record_op("Rnn_relu_cell", [x] + list(args), kwargs)


@register_op("ai.onnx", "Rnn_tanh")
def rnn_tanh(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rnn_tanh."""
    return record_op("Rnn_tanh", [x] + list(args), kwargs)


@register_op("ai.onnx", "Rnn_tanh_cell")
def rnn_tanh_cell(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rnn_tanh_cell."""
    return record_op("Rnn_tanh_cell", [x] + list(args), kwargs)


@register_op("ai.onnx", "Roll")
def roll(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Roll."""
    return record_op("Roll", [x] + list(args), kwargs)


@register_op("ai.onnx", "Rot90")
def rot90(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rot90."""
    return record_op("Rot90", [x] + list(args), kwargs)


@register_op("ai.onnx", "Round_")
def round_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Round_."""
    return record_op("Round_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Row_indices_copy")
def row_indices_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Row_indices_copy."""
    return record_op("Row_indices_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "Row_stack")
def row_stack(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Row_stack."""
    return record_op("Row_stack", [x] + list(args), kwargs)


@register_op("ai.onnx", "Rrelu")
def rrelu(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rrelu."""
    return record_op("Rrelu", [x] + list(args), kwargs)


@register_op("ai.onnx", "Rrelu_")
def rrelu_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rrelu_."""
    return record_op("Rrelu_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Rsqrt")
def rsqrt(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rsqrt."""
    return record_op("Rsqrt", [x] + list(args), kwargs)


@register_op("ai.onnx", "Rsqrt_")
def rsqrt_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rsqrt_."""
    return record_op("Rsqrt_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Rsub")
def rsub(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Rsub."""
    return record_op("Rsub", [x] + list(args), kwargs)


@register_op("ai.onnx", "Saddmm")
def saddmm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Saddmm."""
    return record_op("Saddmm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Scalar_tensor")
def scalar_tensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scalar_tensor."""
    return record_op("Scalar_tensor", [x] + list(args), kwargs)


@register_op("ai.onnx", "Scatter_add")
def scatter_add(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scatter_add."""
    return record_op("Scatter_add", [x] + list(args), kwargs)


@register_op("ai.onnx", "Scatter_reduce")
def scatter_reduce(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Scatter_reduce."""
    return record_op("Scatter_reduce", [x] + list(args), kwargs)


@register_op("ai.onnx", "Searchsorted")
def searchsorted(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Searchsorted."""
    return record_op("Searchsorted", [x] + list(args), kwargs)


@register_op("ai.onnx", "Select")
def select(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Select."""
    return record_op("Select", [x] + list(args), kwargs)


@register_op("ai.onnx", "Select_copy")
def select_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Select_copy."""
    return record_op("Select_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "Select_scatter")
def select_scatter(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Select_scatter."""
    return record_op("Select_scatter", [x] + list(args), kwargs)


@register_op("ai.onnx", "Selu_")
def selu_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Selu_."""
    return record_op("Selu_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sgn")
def sgn(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sgn."""
    return record_op("Sgn", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sigmoid_")
def sigmoid_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sigmoid_."""
    return record_op("Sigmoid_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Signbit")
def signbit(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Signbit."""
    return record_op("Signbit", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sin_")
def sin_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sin_."""
    return record_op("Sin_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sinc")
def sinc(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sinc."""
    return record_op("Sinc", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sinc_")
def sinc_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sinc_."""
    return record_op("Sinc_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sinh_")
def sinh_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sinh_."""
    return record_op("Sinh_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Slice_copy")
def slice_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Slice_copy."""
    return record_op("Slice_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "Slice_inverse")
def slice_inverse(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Slice_inverse."""
    return record_op("Slice_inverse", [x] + list(args), kwargs)


@register_op("ai.onnx", "Slice_scatter")
def slice_scatter(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Slice_scatter."""
    return record_op("Slice_scatter", [x] + list(args), kwargs)


@register_op("ai.onnx", "Slogdet")
def slogdet(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Slogdet."""
    return record_op("Slogdet", [x] + list(args), kwargs)


@register_op("ai.onnx", "Smm")
def smm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Smm."""
    return record_op("Smm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sort")
def sort(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sort."""
    return record_op("Sort", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sparse_bsc_tensor")
def sparse_bsc_tensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sparse_bsc_tensor."""
    return record_op("Sparse_bsc_tensor", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sparse_bsr_tensor")
def sparse_bsr_tensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sparse_bsr_tensor."""
    return record_op("Sparse_bsr_tensor", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sparse_compressed_tensor")
def sparse_compressed_tensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sparse_compressed_tensor."""
    return record_op("Sparse_compressed_tensor", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sparse_coo_tensor")
def sparse_coo_tensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sparse_coo_tensor."""
    return record_op("Sparse_coo_tensor", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sparse_csc_tensor")
def sparse_csc_tensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sparse_csc_tensor."""
    return record_op("Sparse_csc_tensor", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sparse_csr_tensor")
def sparse_csr_tensor(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sparse_csr_tensor."""
    return record_op("Sparse_csr_tensor", [x] + list(args), kwargs)


@register_op("ai.onnx", "Split_copy")
def split_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Split_copy."""
    return record_op("Split_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "Split_with_sizes")
def split_with_sizes(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Split_with_sizes."""
    return record_op("Split_with_sizes", [x] + list(args), kwargs)


@register_op("ai.onnx", "Split_with_sizes_copy")
def split_with_sizes_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Split_with_sizes_copy."""
    return record_op("Split_with_sizes_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "Spmm")
def spmm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Spmm."""
    return record_op("Spmm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sqrt")
def sqrt(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sqrt."""
    return record_op("Sqrt", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sqrt_")
def sqrt_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sqrt_."""
    return record_op("Sqrt_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Square")
def square(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Square."""
    return record_op("Square", [x] + list(args), kwargs)


@register_op("ai.onnx", "Square_")
def square_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Square_."""
    return record_op("Square_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Squeeze")
def squeeze(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Squeeze."""
    return record_op("Squeeze", [x] + list(args), kwargs)


@register_op("ai.onnx", "Squeeze_copy")
def squeeze_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Squeeze_copy."""
    return record_op("Squeeze_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sspaddmm")
def sspaddmm(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sspaddmm."""
    return record_op("Sspaddmm", [x] + list(args), kwargs)


@register_op("ai.onnx", "Std")
def std(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Std."""
    return record_op("Std", [x] + list(args), kwargs)


@register_op("ai.onnx", "Std_mean")
def std_mean(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Std_mean."""
    return record_op("Std_mean", [x] + list(args), kwargs)


@register_op("ai.onnx", "Stft")
def stft(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Stft."""
    return record_op("Stft", [x] + list(args), kwargs)


@register_op("ai.onnx", "Subtract")
def subtract(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Subtract."""
    return record_op("Subtract", [x] + list(args), kwargs)


@register_op("ai.onnx", "Svd")
def svd(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Svd."""
    return record_op("Svd", [x] + list(args), kwargs)


@register_op("ai.onnx", "Swapaxes")
def swapaxes(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Swapaxes."""
    return record_op("Swapaxes", [x] + list(args), kwargs)


@register_op("ai.onnx", "Swapdims")
def swapdims(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Swapdims."""
    return record_op("Swapdims", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sym_constrain_range")
def sym_constrain_range(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sym_constrain_range."""
    return record_op("Sym_constrain_range", [x] + list(args), kwargs)


@register_op("ai.onnx", "Sym_constrain_range_for_size")
def sym_constrain_range_for_size(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Sym_constrain_range_for_size."""
    return record_op("Sym_constrain_range_for_size", [x] + list(args), kwargs)


@register_op("ai.onnx", "T")
def t(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute T."""
    return record_op("T", [x] + list(args), kwargs)


@register_op("ai.onnx", "T_copy")
def t_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute T_copy."""
    return record_op("T_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "Take")
def take(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Take."""
    return record_op("Take", [x] + list(args), kwargs)


@register_op("ai.onnx", "Take_along_dim")
def take_along_dim(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Take_along_dim."""
    return record_op("Take_along_dim", [x] + list(args), kwargs)


@register_op("ai.onnx", "Tan_")
def tan_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Tan_."""
    return record_op("Tan_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Tanh_")
def tanh_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Tanh_."""
    return record_op("Tanh_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Tensor_split")
def tensor_split(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Tensor_split."""
    return record_op("Tensor_split", [x] + list(args), kwargs)


@register_op("ai.onnx", "Tensordot")
def tensordot(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Tensordot."""
    return record_op("Tensordot", [x] + list(args), kwargs)


@register_op("ai.onnx", "Threshold")
def threshold(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Threshold."""
    return record_op("Threshold", [x] + list(args), kwargs)


@register_op("ai.onnx", "Threshold_")
def threshold_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Threshold_."""
    return record_op("Threshold_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Transpose_copy")
def transpose_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Transpose_copy."""
    return record_op("Transpose_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "Trapezoid")
def trapezoid(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Trapezoid."""
    return record_op("Trapezoid", [x] + list(args), kwargs)


@register_op("ai.onnx", "Trapz")
def trapz(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Trapz."""
    return record_op("Trapz", [x] + list(args), kwargs)


@register_op("ai.onnx", "Triangular_solve")
def triangular_solve(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Triangular_solve."""
    return record_op("Triangular_solve", [x] + list(args), kwargs)


@register_op("ai.onnx", "Tril")
def tril(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Tril."""
    return record_op("Tril", [x] + list(args), kwargs)


@register_op("ai.onnx", "Tril_indices")
def tril_indices(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Tril_indices."""
    return record_op("Tril_indices", [x] + list(args), kwargs)


@register_op("ai.onnx", "Triplet_margin_loss")
def triplet_margin_loss(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Triplet_margin_loss."""
    return record_op("Triplet_margin_loss", [x] + list(args), kwargs)


@register_op("ai.onnx", "Triu")
def triu(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Triu."""
    return record_op("Triu", [x] + list(args), kwargs)


@register_op("ai.onnx", "Triu_indices")
def triu_indices(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Triu_indices."""
    return record_op("Triu_indices", [x] + list(args), kwargs)


@register_op("ai.onnx", "True_divide")
def true_divide(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute True_divide."""
    return record_op("True_divide", [x] + list(args), kwargs)


@register_op("ai.onnx", "Trunc")
def trunc(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Trunc."""
    return record_op("Trunc", [x] + list(args), kwargs)


@register_op("ai.onnx", "Trunc_")
def trunc_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Trunc_."""
    return record_op("Trunc_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Unbind")
def unbind(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Unbind."""
    return record_op("Unbind", [x] + list(args), kwargs)


@register_op("ai.onnx", "Unbind_copy")
def unbind_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Unbind_copy."""
    return record_op("Unbind_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "Unflatten")
def unflatten(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Unflatten."""
    return record_op("Unflatten", [x] + list(args), kwargs)


@register_op("ai.onnx", "Unfold_copy")
def unfold_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Unfold_copy."""
    return record_op("Unfold_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "Unique_consecutive")
def unique_consecutive(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Unique_consecutive."""
    return record_op("Unique_consecutive", [x] + list(args), kwargs)


@register_op("ai.onnx", "Unsafe_chunk")
def unsafe_chunk(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Unsafe_chunk."""
    return record_op("Unsafe_chunk", [x] + list(args), kwargs)


@register_op("ai.onnx", "Unsafe_split")
def unsafe_split(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Unsafe_split."""
    return record_op("Unsafe_split", [x] + list(args), kwargs)


@register_op("ai.onnx", "Unsafe_split_with_sizes")
def unsafe_split_with_sizes(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Unsafe_split_with_sizes."""
    return record_op("Unsafe_split_with_sizes", [x] + list(args), kwargs)


@register_op("ai.onnx", "Unsqueeze")
def unsqueeze(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Unsqueeze."""
    return record_op("Unsqueeze", [x] + list(args), kwargs)


@register_op("ai.onnx", "Unsqueeze_copy")
def unsqueeze_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Unsqueeze_copy."""
    return record_op("Unsqueeze_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "Values_copy")
def values_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Values_copy."""
    return record_op("Values_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "Vander")
def vander(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Vander."""
    return record_op("Vander", [x] + list(args), kwargs)


@register_op("ai.onnx", "Var")
def var(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Var."""
    return record_op("Var", [x] + list(args), kwargs)


@register_op("ai.onnx", "Var_mean")
def var_mean(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Var_mean."""
    return record_op("Var_mean", [x] + list(args), kwargs)


@register_op("ai.onnx", "Vdot")
def vdot(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Vdot."""
    return record_op("Vdot", [x] + list(args), kwargs)


@register_op("ai.onnx", "View_as_complex")
def view_as_complex(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute View_as_complex."""
    return record_op("View_as_complex", [x] + list(args), kwargs)


@register_op("ai.onnx", "View_as_complex_copy")
def view_as_complex_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute View_as_complex_copy."""
    return record_op("View_as_complex_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "View_as_real")
def view_as_real(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute View_as_real."""
    return record_op("View_as_real", [x] + list(args), kwargs)


@register_op("ai.onnx", "View_as_real_copy")
def view_as_real_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute View_as_real_copy."""
    return record_op("View_as_real_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "View_copy")
def view_copy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute View_copy."""
    return record_op("View_copy", [x] + list(args), kwargs)


@register_op("ai.onnx", "Vsplit")
def vsplit(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Vsplit."""
    return record_op("Vsplit", [x] + list(args), kwargs)


@register_op("ai.onnx", "Vstack")
def vstack(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Vstack."""
    return record_op("Vstack", [x] + list(args), kwargs)


@register_op("ai.onnx", "Xlogy")
def xlogy(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Xlogy."""
    return record_op("Xlogy", [x] + list(args), kwargs)


@register_op("ai.onnx", "Xlogy_")
def xlogy_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Xlogy_."""
    return record_op("Xlogy_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Zero_")
def zero_(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Zero_."""
    return record_op("Zero_", [x] + list(args), kwargs)


@register_op("ai.onnx", "Zeros_like")
def zeros_like(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Zeros_like."""
    return record_op("Zeros_like", [x] + list(args), kwargs)


@register_op("ai.onnx", "Bfloat16")
def bfloat16(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bfloat16."""
    return record_op("Bfloat16", [x] + list(args), kwargs)


@register_op("ai.onnx", "Bit")
def bit(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bit."""
    return record_op("Bit", [x] + list(args), kwargs)


@register_op("ai.onnx", "Bits16")
def bits16(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bits16."""
    return record_op("Bits16", [x] + list(args), kwargs)


@register_op("ai.onnx", "Bits1x8")
def bits1x8(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bits1x8."""
    return record_op("Bits1x8", [x] + list(args), kwargs)


@register_op("ai.onnx", "Bits2x4")
def bits2x4(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bits2x4."""
    return record_op("Bits2x4", [x] + list(args), kwargs)


@register_op("ai.onnx", "Bits4x2")
def bits4x2(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bits4x2."""
    return record_op("Bits4x2", [x] + list(args), kwargs)


@register_op("ai.onnx", "Bits8")
def bits8(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Bits8."""
    return record_op("Bits8", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cdouble")
def cdouble(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cdouble."""
    return record_op("Cdouble", [x] + list(args), kwargs)


@register_op("ai.onnx", "Cfloat")
def cfloat(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Cfloat."""
    return record_op("Cfloat", [x] + list(args), kwargs)


@register_op("ai.onnx", "Chalf")
def chalf(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Chalf."""
    return record_op("Chalf", [x] + list(args), kwargs)


@register_op("ai.onnx", "Complex128")
def complex128(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Complex128."""
    return record_op("Complex128", [x] + list(args), kwargs)


@register_op("ai.onnx", "Complex32")
def complex32(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Complex32."""
    return record_op("Complex32", [x] + list(args), kwargs)


@register_op("ai.onnx", "Complex64")
def complex64(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Complex64."""
    return record_op("Complex64", [x] + list(args), kwargs)


@register_op("ai.onnx", "Double")
def double(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Double."""
    return record_op("Double", [x] + list(args), kwargs)


@register_op("ai.onnx", "Float")
def float(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Float."""
    return record_op("Float", [x] + list(args), kwargs)


@register_op("ai.onnx", "Float16")
def float16(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Float16."""
    return record_op("Float16", [x] + list(args), kwargs)


@register_op("ai.onnx", "Float4_e2m1fn_x2")
def float4_e2m1fn_x2(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Float4_e2m1fn_x2."""
    return record_op("Float4_e2m1fn_x2", [x] + list(args), kwargs)


@register_op("ai.onnx", "Float8_e4m3fn")
def float8_e4m3fn(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Float8_e4m3fn."""
    return record_op("Float8_e4m3fn", [x] + list(args), kwargs)


@register_op("ai.onnx", "Float8_e4m3fnuz")
def float8_e4m3fnuz(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Float8_e4m3fnuz."""
    return record_op("Float8_e4m3fnuz", [x] + list(args), kwargs)


@register_op("ai.onnx", "Float8_e5m2")
def float8_e5m2(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Float8_e5m2."""
    return record_op("Float8_e5m2", [x] + list(args), kwargs)


@register_op("ai.onnx", "Float8_e5m2fnuz")
def float8_e5m2fnuz(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Float8_e5m2fnuz."""
    return record_op("Float8_e5m2fnuz", [x] + list(args), kwargs)


@register_op("ai.onnx", "Float8_e8m0fnu")
def float8_e8m0fnu(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Float8_e8m0fnu."""
    return record_op("Float8_e8m0fnu", [x] + list(args), kwargs)


@register_op("ai.onnx", "Half")
def half(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Half."""
    return record_op("Half", [x] + list(args), kwargs)


@register_op("ai.onnx", "Int")
def int(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Int."""
    return record_op("Int", [x] + list(args), kwargs)


@register_op("ai.onnx", "Int1")
def int1(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Int1."""
    return record_op("Int1", [x] + list(args), kwargs)


@register_op("ai.onnx", "Int16")
def int16(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Int16."""
    return record_op("Int16", [x] + list(args), kwargs)


@register_op("ai.onnx", "Int2")
def int2(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Int2."""
    return record_op("Int2", [x] + list(args), kwargs)


@register_op("ai.onnx", "Int3")
def int3(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Int3."""
    return record_op("Int3", [x] + list(args), kwargs)


@register_op("ai.onnx", "Int4")
def int4(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Int4."""
    return record_op("Int4", [x] + list(args), kwargs)


@register_op("ai.onnx", "Int5")
def int5(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Int5."""
    return record_op("Int5", [x] + list(args), kwargs)


@register_op("ai.onnx", "Int6")
def int6(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Int6."""
    return record_op("Int6", [x] + list(args), kwargs)


@register_op("ai.onnx", "Int7")
def int7(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Int7."""
    return record_op("Int7", [x] + list(args), kwargs)


@register_op("ai.onnx", "Int8")
def int8(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Int8."""
    return record_op("Int8", [x] + list(args), kwargs)


@register_op("ai.onnx", "Long")
def long(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Long."""
    return record_op("Long", [x] + list(args), kwargs)


@register_op("ai.onnx", "Qint32")
def qint32(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Qint32."""
    return record_op("Qint32", [x] + list(args), kwargs)


@register_op("ai.onnx", "Qint8")
def qint8(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Qint8."""
    return record_op("Qint8", [x] + list(args), kwargs)


@register_op("ai.onnx", "Quint2x4")
def quint2x4(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quint2x4."""
    return record_op("Quint2x4", [x] + list(args), kwargs)


@register_op("ai.onnx", "Quint4x2")
def quint4x2(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quint4x2."""
    return record_op("Quint4x2", [x] + list(args), kwargs)


@register_op("ai.onnx", "Quint8")
def quint8(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Quint8."""
    return record_op("Quint8", [x] + list(args), kwargs)


@register_op("ai.onnx", "Short")
def short(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Short."""
    return record_op("Short", [x] + list(args), kwargs)


@register_op("ai.onnx", "Uint1")
def uint1(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Uint1."""
    return record_op("Uint1", [x] + list(args), kwargs)


@register_op("ai.onnx", "Uint16")
def uint16(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Uint16."""
    return record_op("Uint16", [x] + list(args), kwargs)


@register_op("ai.onnx", "Uint2")
def uint2(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Uint2."""
    return record_op("Uint2", [x] + list(args), kwargs)


@register_op("ai.onnx", "Uint3")
def uint3(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Uint3."""
    return record_op("Uint3", [x] + list(args), kwargs)


@register_op("ai.onnx", "Uint32")
def uint32(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Uint32."""
    return record_op("Uint32", [x] + list(args), kwargs)


@register_op("ai.onnx", "Uint4")
def uint4(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Uint4."""
    return record_op("Uint4", [x] + list(args), kwargs)


@register_op("ai.onnx", "Uint5")
def uint5(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Uint5."""
    return record_op("Uint5", [x] + list(args), kwargs)


@register_op("ai.onnx", "Uint6")
def uint6(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Uint6."""
    return record_op("Uint6", [x] + list(args), kwargs)


@register_op("ai.onnx", "Uint64")
def uint64(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Uint64."""
    return record_op("Uint64", [x] + list(args), kwargs)


@register_op("ai.onnx", "Uint7")
def uint7(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Uint7."""
    return record_op("Uint7", [x] + list(args), kwargs)


@register_op("ai.onnx", "Uint8")
def uint8(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    """Compute Uint8."""
    return record_op("Uint8", [x] + list(args), kwargs)
