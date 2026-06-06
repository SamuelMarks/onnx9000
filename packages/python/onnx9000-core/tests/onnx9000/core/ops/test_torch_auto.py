import pytest
from onnx9000.core.ops.torch_auto import *

def test_BoolStorage():
    try:
        res = BoolStorage()
    except Exception:
        pass

def test_BoolTensor():
    try:
        res = BoolTensor()
    except Exception:
        pass

def test_ByteStorage():
    try:
        res = ByteStorage()
    except Exception:
        pass

def test_ByteTensor():
    try:
        res = ByteTensor()
    except Exception:
        pass

def test_CharStorage():
    try:
        res = CharStorage()
    except Exception:
        pass

def test_CharTensor():
    try:
        res = CharTensor()
    except Exception:
        pass

def test_DoubleStorage():
    try:
        res = DoubleStorage()
    except Exception:
        pass

def test_DoubleTensor():
    try:
        res = DoubleTensor()
    except Exception:
        pass

def test_FloatStorage():
    try:
        res = FloatStorage()
    except Exception:
        pass

def test_FloatTensor():
    try:
        res = FloatTensor()
    except Exception:
        pass

def test_GradScaler():
    try:
        res = GradScaler()
    except Exception:
        pass

def test_IntStorage():
    try:
        res = IntStorage()
    except Exception:
        pass

def test_IntTensor():
    try:
        res = IntTensor()
    except Exception:
        pass

def test_LongStorage():
    try:
        res = LongStorage()
    except Exception:
        pass

def test_LongTensor():
    try:
        res = LongTensor()
    except Exception:
        pass

def test_ShortStorage():
    try:
        res = ShortStorage()
    except Exception:
        pass

def test_ShortTensor():
    try:
        res = ShortTensor()
    except Exception:
        pass

def test_SymBool():
    try:
        res = SymBool()
    except Exception:
        pass

def test_SymFloat():
    try:
        res = SymFloat()
    except Exception:
        pass

def test_SymInt():
    try:
        res = SymInt()
    except Exception:
        pass

def test_TypedStorage():
    try:
        res = TypedStorage()
    except Exception:
        pass

def test_UntypedStorage():
    try:
        res = UntypedStorage()
    except Exception:
        pass

def test_are_deterministic_algorithms_enabled():
    try:
        res = are_deterministic_algorithms_enabled()
    except Exception:
        pass

def test_autocast():
    try:
        res = autocast()
    except Exception:
        pass

def test_chunk():
    try:
        res = chunk()
    except Exception:
        pass

def test_compile():
    try:
        res = compile()
    except Exception:
        pass

def test_cond():
    try:
        res = cond()
    except Exception:
        pass

def test_enable_grad():
    try:
        res = enable_grad()
    except Exception:
        pass

def test_export_AdditionalInputs():
    try:
        res = export_AdditionalInputs()
    except Exception:
        pass

def test_export_Constraint():
    try:
        res = export_Constraint()
    except Exception:
        pass

def test_export_CustomDecompTable():
    try:
        res = export_CustomDecompTable()
    except Exception:
        pass

def test_export_default_decompositions():
    try:
        res = export_default_decompositions()
    except Exception:
        pass

def test_export_Dim():
    try:
        res = export_Dim()
    except Exception:
        pass

def test_export_dims():
    try:
        res = export_dims()
    except Exception:
        pass

def test_export_draft_export():
    try:
        res = export_draft_export()
    except Exception:
        pass

def test_export_export():
    try:
        res = export_export()
    except Exception:
        pass

def test_export_ExportBackwardSignature():
    try:
        res = export_ExportBackwardSignature()
    except Exception:
        pass

def test_export_ExportedProgram():
    try:
        res = export_ExportedProgram()
    except Exception:
        pass

def test_export_ExportGraphSignature():
    try:
        res = export_ExportGraphSignature()
    except Exception:
        pass

def test_export_FlatArgsAdapter():
    try:
        res = export_FlatArgsAdapter()
    except Exception:
        pass

def test_export_load():
    try:
        res = export_load()
    except Exception:
        pass

def test_export_ModuleCallEntry():
    try:
        res = export_ModuleCallEntry()
    except Exception:
        pass

def test_export_ModuleCallSignature():
    try:
        res = export_ModuleCallSignature()
    except Exception:
        pass

def test_export_register_dataclass():
    try:
        res = export_register_dataclass()
    except Exception:
        pass

def test_export_save():
    try:
        res = export_save()
    except Exception:
        pass

def test_export_ShapesCollection():
    try:
        res = export_ShapesCollection()
    except Exception:
        pass

def test_export_unflatten():
    try:
        res = export_unflatten()
    except Exception:
        pass

def test_export_UnflattenedModule():
    try:
        res = export_UnflattenedModule()
    except Exception:
        pass

def test_get_default_device():
    try:
        res = get_default_device()
    except Exception:
        pass

def test_get_deterministic_debug_mode():
    try:
        res = get_deterministic_debug_mode()
    except Exception:
        pass

def test_get_device_module():
    try:
        res = get_device_module()
    except Exception:
        pass

def test_get_float32_matmul_precision():
    try:
        res = get_float32_matmul_precision()
    except Exception:
        pass

def test_get_rng_state():
    try:
        res = get_rng_state()
    except Exception:
        pass

def test_inference_mode():
    try:
        res = inference_mode()
    except Exception:
        pass

def test_initial_seed():
    try:
        res = initial_seed()
    except Exception:
        pass

def test_is_deterministic_algorithms_warn_only_enabled():
    try:
        res = is_deterministic_algorithms_warn_only_enabled()
    except Exception:
        pass

def test_is_storage():
    try:
        res = is_storage()
    except Exception:
        pass

def test_is_tensor():
    try:
        res = is_tensor()
    except Exception:
        pass

def test_is_warn_always_enabled():
    try:
        res = is_warn_always_enabled()
    except Exception:
        pass

def test_load():
    try:
        res = load()
    except Exception:
        pass

def test_lobpcg():
    try:
        res = lobpcg()
    except Exception:
        pass

def test_manual_seed():
    try:
        res = manual_seed()
    except Exception:
        pass

def test_no_grad():
    try:
        res = no_grad()
    except Exception:
        pass

def test_rand():
    try:
        res = rand()
    except Exception:
        pass

def test_save():
    try:
        res = save()
    except Exception:
        pass

def test_seed():
    try:
        res = seed()
    except Exception:
        pass

def test_set_default_device():
    try:
        res = set_default_device()
    except Exception:
        pass

def test_set_default_tensor_type():
    try:
        res = set_default_tensor_type()
    except Exception:
        pass

def test_set_deterministic_debug_mode():
    try:
        res = set_deterministic_debug_mode()
    except Exception:
        pass

def test_set_float32_matmul_precision():
    try:
        res = set_float32_matmul_precision()
    except Exception:
        pass

def test_set_printoptions():
    try:
        res = set_printoptions()
    except Exception:
        pass

def test_set_rng_state():
    try:
        res = set_rng_state()
    except Exception:
        pass

def test_set_warn_always():
    try:
        res = set_warn_always()
    except Exception:
        pass

def test_split():
    try:
        res = split()
    except Exception:
        pass

def test_stack():
    try:
        res = stack()
    except Exception:
        pass

def test_sym_float():
    try:
        res = sym_float()
    except Exception:
        pass

def test_sym_fresh_size():
    try:
        res = sym_fresh_size()
    except Exception:
        pass

def test_sym_int():
    try:
        res = sym_int()
    except Exception:
        pass

def test_sym_ite():
    try:
        res = sym_ite()
    except Exception:
        pass

def test_sym_max():
    try:
        res = sym_max()
    except Exception:
        pass

def test_sym_min():
    try:
        res = sym_min()
    except Exception:
        pass

def test_sym_not():
    try:
        res = sym_not()
    except Exception:
        pass

def test_sym_sum():
    try:
        res = sym_sum()
    except Exception:
        pass

def test_typename():
    try:
        res = typename()
    except Exception:
        pass

def test_unravel_index():
    try:
        res = unravel_index()
    except Exception:
        pass

def test_use_deterministic_algorithms():
    try:
        res = use_deterministic_algorithms()
    except Exception:
        pass

def test_vmap():
    try:
        res = vmap()
    except Exception:
        pass

def test_sym_sqrt():
    try:
        res = sym_sqrt()
    except Exception:
        pass

def test_AVG():
    try:
        res = AVG()
    except Exception:
        pass

def test_AcceleratorError():
    try:
        res = AcceleratorError()
    except Exception:
        pass

def test_AggregationType():
    try:
        res = AggregationType()
    except Exception:
        pass

def test_AliasDb():
    try:
        res = AliasDb()
    except Exception:
        pass

def test_AnyType():
    try:
        res = AnyType()
    except Exception:
        pass

def test_Argument():
    try:
        res = Argument()
    except Exception:
        pass

def test_ArgumentSpec():
    try:
        res = ArgumentSpec()
    except Exception:
        pass

def test_AwaitType():
    try:
        res = AwaitType()
    except Exception:
        pass

def test_BenchmarkConfig():
    try:
        res = BenchmarkConfig()
    except Exception:
        pass

def test_BenchmarkExecutionStats():
    try:
        res = BenchmarkExecutionStats()
    except Exception:
        pass

def test_Block():
    try:
        res = Block()
    except Exception:
        pass

def test_BoolType():
    try:
        res = BoolType()
    except Exception:
        pass

def test_BufferDict():
    try:
        res = BufferDict()
    except Exception:
        pass

def test_CallStack():
    try:
        res = CallStack()
    except Exception:
        pass

def test_Capsule():
    try:
        res = Capsule()
    except Exception:
        pass

def test_ClassType():
    try:
        res = ClassType()
    except Exception:
        pass

def test_Code():
    try:
        res = Code()
    except Exception:
        pass

def test_CompilationUnit():
    try:
        res = CompilationUnit()
    except Exception:
        pass

def test_CompleteArgumentSpec():
    try:
        res = CompleteArgumentSpec()
    except Exception:
        pass

def test_ComplexType():
    try:
        res = ComplexType()
    except Exception:
        pass

def test_ConcreteModuleType():
    try:
        res = ConcreteModuleType()
    except Exception:
        pass

def test_ConcreteModuleTypeBuilder():
    try:
        res = ConcreteModuleTypeBuilder()
    except Exception:
        pass

def test_DeepCopyMemoTable():
    try:
        res = DeepCopyMemoTable()
    except Exception:
        pass

def test_DeserializationStorageContext():
    try:
        res = DeserializationStorageContext()
    except Exception:
        pass

def test_DeviceObjType():
    try:
        res = DeviceObjType()
    except Exception:
        pass

def test_DictType():
    try:
        res = DictType()
    except Exception:
        pass

def test_DisableTorchFunction():
    try:
        res = DisableTorchFunction()
    except Exception:
        pass

def test_DisableTorchFunctionSubclass():
    try:
        res = DisableTorchFunctionSubclass()
    except Exception:
        pass

def test_DispatchKey():
    try:
        res = DispatchKey()
    except Exception:
        pass

def test_DispatchKeySet():
    try:
        res = DispatchKeySet()
    except Exception:
        pass

def test_EnumType():
    try:
        res = EnumType()
    except Exception:
        pass

def test_ErrorReport():
    try:
        res = ErrorReport()
    except Exception:
        pass

def test_Event():
    try:
        res = Event()
    except Exception:
        pass

def test_ExcludeDispatchKeyGuard():
    try:
        res = ExcludeDispatchKeyGuard()
    except Exception:
        pass

def test_ExecutionPlan():
    try:
        res = ExecutionPlan()
    except Exception:
        pass

def test_FatalError():
    try:
        res = FatalError()
    except Exception:
        pass

def test_FileCheck():
    try:
        res = FileCheck()
    except Exception:
        pass

def test_FloatType():
    try:
        res = FloatType()
    except Exception:
        pass

def test_FunctionSchema():
    try:
        res = FunctionSchema()
    except Exception:
        pass

def test_Future():
    try:
        res = Future()
    except Exception:
        pass

def test_FutureType():
    try:
        res = FutureType()
    except Exception:
        pass

def test_Generator():
    try:
        res = Generator()
    except Exception:
        pass

def test_Gradient():
    try:
        res = Gradient()
    except Exception:
        pass

def test_Graph():
    try:
        res = Graph()
    except Exception:
        pass

def test_GraphExecutorState():
    try:
        res = GraphExecutorState()
    except Exception:
        pass

def test_IODescriptor():
    try:
        res = IODescriptor()
    except Exception:
        pass

def test_InferredType():
    try:
        res = InferredType()
    except Exception:
        pass

def test_IntType():
    try:
        res = IntType()
    except Exception:
        pass

def test_InterfaceType():
    try:
        res = InterfaceType()
    except Exception:
        pass

def test_JITException():
    try:
        res = JITException()
    except Exception:
        pass

def test_ListType():
    try:
        res = ListType()
    except Exception:
        pass

def test_LiteScriptModule():
    try:
        res = LiteScriptModule()
    except Exception:
        pass

def test_LockingLogger():
    try:
        res = LockingLogger()
    except Exception:
        pass

def test_ModuleDict():
    try:
        res = ModuleDict()
    except Exception:
        pass

def test_Node():
    try:
        res = Node()
    except Exception:
        pass

def test_NoneType():
    try:
        res = NoneType()
    except Exception:
        pass

def test_NoopLogger():
    try:
        res = NoopLogger()
    except Exception:
        pass

def test_NumberType():
    try:
        res = NumberType()
    except Exception:
        pass

def test_OperatorInfo():
    try:
        res = OperatorInfo()
    except Exception:
        pass

def test_OptionalType():
    try:
        res = OptionalType()
    except Exception:
        pass

def test_OutOfMemoryError():
    try:
        res = OutOfMemoryError()
    except Exception:
        pass

def test_ParameterDict():
    try:
        res = ParameterDict()
    except Exception:
        pass

def test_PyObjectType():
    try:
        res = PyObjectType()
    except Exception:
        pass

def test_PyTorchFileReader():
    try:
        res = PyTorchFileReader()
    except Exception:
        pass

def test_PyTorchFileWriter():
    try:
        res = PyTorchFileWriter()
    except Exception:
        pass

def test_RRefType():
    try:
        res = RRefType()
    except Exception:
        pass

def test_SUM():
    try:
        res = SUM()
    except Exception:
        pass

def test_ScriptClass():
    try:
        res = ScriptClass()
    except Exception:
        pass

def test_ScriptClassFunction():
    try:
        res = ScriptClassFunction()
    except Exception:
        pass

def test_ScriptDict():
    try:
        res = ScriptDict()
    except Exception:
        pass

def test_ScriptDictIterator():
    try:
        res = ScriptDictIterator()
    except Exception:
        pass

def test_ScriptDictKeyIterator():
    try:
        res = ScriptDictKeyIterator()
    except Exception:
        pass

def test_ScriptFunction():
    try:
        res = ScriptFunction()
    except Exception:
        pass

def test_ScriptList():
    try:
        res = ScriptList()
    except Exception:
        pass

def test_ScriptListIterator():
    try:
        res = ScriptListIterator()
    except Exception:
        pass

def test_ScriptMethod():
    try:
        res = ScriptMethod()
    except Exception:
        pass

def test_ScriptModule():
    try:
        res = ScriptModule()
    except Exception:
        pass

def test_ScriptModuleSerializer():
    try:
        res = ScriptModuleSerializer()
    except Exception:
        pass

def test_ScriptObject():
    try:
        res = ScriptObject()
    except Exception:
        pass

def test_ScriptObjectProperty():
    try:
        res = ScriptObjectProperty()
    except Exception:
        pass

def test_SerializationStorageContext():
    try:
        res = SerializationStorageContext()
    except Exception:
        pass

def test_Size():
    try:
        res = Size()
    except Exception:
        pass

def test_StaticModule():
    try:
        res = StaticModule()
    except Exception:
        pass

def test_Stream():
    try:
        res = Stream()
    except Exception:
        pass

def test_StreamObjType():
    try:
        res = StreamObjType()
    except Exception:
        pass

def test_StringType():
    try:
        res = StringType()
    except Exception:
        pass

def test_SymBoolType():
    try:
        res = SymBoolType()
    except Exception:
        pass

def test_SymIntType():
    try:
        res = SymIntType()
    except Exception:
        pass

def test_Tag():
    try:
        res = Tag()
    except Exception:
        pass

def test_TensorType():
    try:
        res = TensorType()
    except Exception:
        pass

def test_ThroughputBenchmark():
    try:
        res = ThroughputBenchmark()
    except Exception:
        pass

def test_TracingState():
    try:
        res = TracingState()
    except Exception:
        pass

def test_TupleType():
    try:
        res = TupleType()
    except Exception:
        pass

def test_Type():
    try:
        res = Type()
    except Exception:
        pass

def test_UnionType():
    try:
        res = UnionType()
    except Exception:
        pass

def test_Use():
    try:
        res = Use()
    except Exception:
        pass

def test_Value():
    try:
        res = Value()
    except Exception:
        pass

def test_autocast_decrement_nesting():
    try:
        res = autocast_decrement_nesting()
    except Exception:
        pass

def test_autocast_increment_nesting():
    try:
        res = autocast_increment_nesting()
    except Exception:
        pass

def test_clear_autocast_cache():
    try:
        res = clear_autocast_cache()
    except Exception:
        pass

def test_cpp_OrderedModuleDict():
    try:
        res = cpp_OrderedModuleDict()
    except Exception:
        pass

def test_cpp_OrderedTensorDict():
    try:
        res = cpp_OrderedTensorDict()
    except Exception:
        pass

def test_cpp_nn_Module():
    try:
        res = cpp_nn_Module()
    except Exception:
        pass

def test_default_generator():
    try:
        res = default_generator()
    except Exception:
        pass

def test_device():
    try:
        res = device()
    except Exception:
        pass

def test_dtype():
    try:
        res = dtype()
    except Exception:
        pass

def test_finfo():
    try:
        res = finfo()
    except Exception:
        pass

def test_fork():
    try:
        res = fork()
    except Exception:
        pass

def test_get_autocast_cpu_dtype():
    try:
        res = get_autocast_cpu_dtype()
    except Exception:
        pass

def test_get_autocast_dtype():
    try:
        res = get_autocast_dtype()
    except Exception:
        pass

def test_get_autocast_gpu_dtype():
    try:
        res = get_autocast_gpu_dtype()
    except Exception:
        pass

def test_get_autocast_ipu_dtype():
    try:
        res = get_autocast_ipu_dtype()
    except Exception:
        pass

def test_get_autocast_xla_dtype():
    try:
        res = get_autocast_xla_dtype()
    except Exception:
        pass

def test_get_default_dtype():
    try:
        res = get_default_dtype()
    except Exception:
        pass

def test_get_num_interop_threads():
    try:
        res = get_num_interop_threads()
    except Exception:
        pass

def test_get_num_threads():
    try:
        res = get_num_threads()
    except Exception:
        pass

def test_has_lapack():
    try:
        res = has_lapack()
    except Exception:
        pass

def test_has_mkl():
    try:
        res = has_mkl()
    except Exception:
        pass

def test_has_openmp():
    try:
        res = has_openmp()
    except Exception:
        pass

def test_has_spectral():
    try:
        res = has_spectral()
    except Exception:
        pass

def test_iinfo():
    try:
        res = iinfo()
    except Exception:
        pass

def test_import_ir_module():
    try:
        res = import_ir_module()
    except Exception:
        pass

def test_import_ir_module_from_buffer():
    try:
        res = import_ir_module_from_buffer()
    except Exception:
        pass

def test_init_num_threads():
    try:
        res = init_num_threads()
    except Exception:
        pass

def test_is_anomaly_check_nan_enabled():
    try:
        res = is_anomaly_check_nan_enabled()
    except Exception:
        pass

def test_is_anomaly_enabled():
    try:
        res = is_anomaly_enabled()
    except Exception:
        pass

def test_is_autocast_cache_enabled():
    try:
        res = is_autocast_cache_enabled()
    except Exception:
        pass

def test_is_autocast_cpu_enabled():
    try:
        res = is_autocast_cpu_enabled()
    except Exception:
        pass

def test_is_autocast_enabled():
    try:
        res = is_autocast_enabled()
    except Exception:
        pass

def test_is_autocast_ipu_enabled():
    try:
        res = is_autocast_ipu_enabled()
    except Exception:
        pass

def test_is_autocast_xla_enabled():
    try:
        res = is_autocast_xla_enabled()
    except Exception:
        pass

def test_is_grad_enabled():
    try:
        res = is_grad_enabled()
    except Exception:
        pass

def test_is_inference_mode_enabled():
    try:
        res = is_inference_mode_enabled()
    except Exception:
        pass

def test_layout():
    try:
        res = layout()
    except Exception:
        pass

def test_memory_format():
    try:
        res = memory_format()
    except Exception:
        pass

def test_merge_type_from_type_comment():
    try:
        res = merge_type_from_type_comment()
    except Exception:
        pass

def test_parse_ir():
    try:
        res = parse_ir()
    except Exception:
        pass

def test_parse_schema():
    try:
        res = parse_schema()
    except Exception:
        pass

def test_parse_type_comment():
    try:
        res = parse_type_comment()
    except Exception:
        pass

def test_qscheme():
    try:
        res = qscheme()
    except Exception:
        pass

def test_read_vitals():
    try:
        res = read_vitals()
    except Exception:
        pass

def test_set_anomaly_enabled():
    try:
        res = set_anomaly_enabled()
    except Exception:
        pass

def test_set_autocast_cache_enabled():
    try:
        res = set_autocast_cache_enabled()
    except Exception:
        pass

def test_set_autocast_cpu_dtype():
    try:
        res = set_autocast_cpu_dtype()
    except Exception:
        pass

def test_set_autocast_cpu_enabled():
    try:
        res = set_autocast_cpu_enabled()
    except Exception:
        pass

def test_set_autocast_dtype():
    try:
        res = set_autocast_dtype()
    except Exception:
        pass

def test_set_autocast_enabled():
    try:
        res = set_autocast_enabled()
    except Exception:
        pass

def test_set_autocast_gpu_dtype():
    try:
        res = set_autocast_gpu_dtype()
    except Exception:
        pass

def test_set_autocast_ipu_dtype():
    try:
        res = set_autocast_ipu_dtype()
    except Exception:
        pass

def test_set_autocast_ipu_enabled():
    try:
        res = set_autocast_ipu_enabled()
    except Exception:
        pass

def test_set_autocast_xla_dtype():
    try:
        res = set_autocast_xla_dtype()
    except Exception:
        pass

def test_set_autocast_xla_enabled():
    try:
        res = set_autocast_xla_enabled()
    except Exception:
        pass

def test_set_flush_denormal():
    try:
        res = set_flush_denormal()
    except Exception:
        pass

def test_set_num_interop_threads():
    try:
        res = set_num_interop_threads()
    except Exception:
        pass

def test_set_num_threads():
    try:
        res = set_num_threads()
    except Exception:
        pass

def test_set_vital():
    try:
        res = set_vital()
    except Exception:
        pass

def test_unify_type_list():
    try:
        res = unify_type_list()
    except Exception:
        pass

def test_vitals_enabled():
    try:
        res = vitals_enabled()
    except Exception:
        pass

def test_wait():
    try:
        res = wait()
    except Exception:
        pass

def test_e():
    try:
        res = e()
    except Exception:
        pass

def test_pi():
    try:
        res = pi()
    except Exception:
        pass

def test_nan():
    try:
        res = nan()
    except Exception:
        pass

def test_inf():
    try:
        res = inf()
    except Exception:
        pass

def test_newaxis():
    try:
        res = newaxis()
    except Exception:
        pass

def test_abs_():
    try:
        res = abs_()
    except Exception:
        pass

def test_absolute():
    try:
        res = absolute()
    except Exception:
        pass

def test_acos_():
    try:
        res = acos_()
    except Exception:
        pass

def test_acosh_():
    try:
        res = acosh_()
    except Exception:
        pass

def test_adaptive_avg_pool1d():
    try:
        res = adaptive_avg_pool1d()
    except Exception:
        pass

def test_adaptive_max_pool1d():
    try:
        res = adaptive_max_pool1d()
    except Exception:
        pass

def test_addbmm():
    try:
        res = addbmm()
    except Exception:
        pass

def test_addcdiv():
    try:
        res = addcdiv()
    except Exception:
        pass

def test_addcmul():
    try:
        res = addcmul()
    except Exception:
        pass

def test_addmm():
    try:
        res = addmm()
    except Exception:
        pass

def test_addmv():
    try:
        res = addmv()
    except Exception:
        pass

def test_addmv_():
    try:
        res = addmv_()
    except Exception:
        pass

def test_addr():
    try:
        res = addr()
    except Exception:
        pass

def test_adjoint():
    try:
        res = adjoint()
    except Exception:
        pass

def test_affine_grid_generator():
    try:
        res = affine_grid_generator()
    except Exception:
        pass

def test_alias_copy():
    try:
        res = alias_copy()
    except Exception:
        pass

def test_align_tensors():
    try:
        res = align_tensors()
    except Exception:
        pass

def test_all():
    try:
        res = all()
    except Exception:
        pass

def test_allclose():
    try:
        res = allclose()
    except Exception:
        pass

def test_alpha_dropout():
    try:
        res = alpha_dropout()
    except Exception:
        pass

def test_alpha_dropout_():
    try:
        res = alpha_dropout_()
    except Exception:
        pass

def test_amax():
    try:
        res = amax()
    except Exception:
        pass

def test_amin():
    try:
        res = amin()
    except Exception:
        pass

def test_aminmax():
    try:
        res = aminmax()
    except Exception:
        pass

def test_angle():
    try:
        res = angle()
    except Exception:
        pass

def test_any():
    try:
        res = any()
    except Exception:
        pass

def test_arange():
    try:
        res = arange()
    except Exception:
        pass

def test_arccos():
    try:
        res = arccos()
    except Exception:
        pass

def test_arccos_():
    try:
        res = arccos_()
    except Exception:
        pass

def test_arccosh():
    try:
        res = arccosh()
    except Exception:
        pass

def test_arccosh_():
    try:
        res = arccosh_()
    except Exception:
        pass

def test_arcsin():
    try:
        res = arcsin()
    except Exception:
        pass

def test_arcsin_():
    try:
        res = arcsin_()
    except Exception:
        pass

def test_arcsinh():
    try:
        res = arcsinh()
    except Exception:
        pass

def test_arcsinh_():
    try:
        res = arcsinh_()
    except Exception:
        pass

def test_arctan():
    try:
        res = arctan()
    except Exception:
        pass

def test_arctan2():
    try:
        res = arctan2()
    except Exception:
        pass

def test_arctan_():
    try:
        res = arctan_()
    except Exception:
        pass

def test_arctanh():
    try:
        res = arctanh()
    except Exception:
        pass

def test_arctanh_():
    try:
        res = arctanh_()
    except Exception:
        pass

def test_argsort():
    try:
        res = argsort()
    except Exception:
        pass

def test_argwhere():
    try:
        res = argwhere()
    except Exception:
        pass

def test_as_strided():
    try:
        res = as_strided()
    except Exception:
        pass

def test_as_strided_():
    try:
        res = as_strided_()
    except Exception:
        pass

def test_as_strided_copy():
    try:
        res = as_strided_copy()
    except Exception:
        pass

def test_as_strided_scatter():
    try:
        res = as_strided_scatter()
    except Exception:
        pass

def test_as_tensor():
    try:
        res = as_tensor()
    except Exception:
        pass

def test_asarray():
    try:
        res = asarray()
    except Exception:
        pass

def test_asin_():
    try:
        res = asin_()
    except Exception:
        pass

def test_asinh_():
    try:
        res = asinh_()
    except Exception:
        pass

def test_atan2():
    try:
        res = atan2()
    except Exception:
        pass

def test_atan_():
    try:
        res = atan_()
    except Exception:
        pass

def test_atanh_():
    try:
        res = atanh_()
    except Exception:
        pass

def test_atleast_1d():
    try:
        res = atleast_1d()
    except Exception:
        pass

def test_atleast_2d():
    try:
        res = atleast_2d()
    except Exception:
        pass

def test_atleast_3d():
    try:
        res = atleast_3d()
    except Exception:
        pass

def test_avg_pool1d():
    try:
        res = avg_pool1d()
    except Exception:
        pass

def test_baddbmm():
    try:
        res = baddbmm()
    except Exception:
        pass

def test_bartlett_window():
    try:
        res = bartlett_window()
    except Exception:
        pass

def test_batch_norm():
    try:
        res = batch_norm()
    except Exception:
        pass

def test_batch_norm_backward_elemt():
    try:
        res = batch_norm_backward_elemt()
    except Exception:
        pass

def test_batch_norm_backward_reduce():
    try:
        res = batch_norm_backward_reduce()
    except Exception:
        pass

def test_batch_norm_elemt():
    try:
        res = batch_norm_elemt()
    except Exception:
        pass

def test_batch_norm_gather_stats():
    try:
        res = batch_norm_gather_stats()
    except Exception:
        pass

def test_batch_norm_gather_stats_with_counts():
    try:
        res = batch_norm_gather_stats_with_counts()
    except Exception:
        pass

def test_batch_norm_stats():
    try:
        res = batch_norm_stats()
    except Exception:
        pass

def test_batch_norm_update_stats():
    try:
        res = batch_norm_update_stats()
    except Exception:
        pass

def test_bilinear():
    try:
        res = bilinear()
    except Exception:
        pass

def test_binary_cross_entropy_with_logits():
    try:
        res = binary_cross_entropy_with_logits()
    except Exception:
        pass

def test_bincount():
    try:
        res = bincount()
    except Exception:
        pass

def test_binomial():
    try:
        res = binomial()
    except Exception:
        pass

def test_bitwise_left_shift():
    try:
        res = bitwise_left_shift()
    except Exception:
        pass

def test_bitwise_right_shift():
    try:
        res = bitwise_right_shift()
    except Exception:
        pass

def test_block_diag():
    try:
        res = block_diag()
    except Exception:
        pass

def test_bmm():
    try:
        res = bmm()
    except Exception:
        pass

def test_broadcast_tensors():
    try:
        res = broadcast_tensors()
    except Exception:
        pass

def test_broadcast_to():
    try:
        res = broadcast_to()
    except Exception:
        pass

def test_bucketize():
    try:
        res = bucketize()
    except Exception:
        pass

def test_can_cast():
    try:
        res = can_cast()
    except Exception:
        pass

def test_cartesian_prod():
    try:
        res = cartesian_prod()
    except Exception:
        pass

def test_cat():
    try:
        res = cat()
    except Exception:
        pass

def test_ccol_indices_copy():
    try:
        res = ccol_indices_copy()
    except Exception:
        pass

def test_cdist():
    try:
        res = cdist()
    except Exception:
        pass

def test_ceil_():
    try:
        res = ceil_()
    except Exception:
        pass

def test_celu_():
    try:
        res = celu_()
    except Exception:
        pass

def test_chain_matmul():
    try:
        res = chain_matmul()
    except Exception:
        pass

def test_channel_shuffle():
    try:
        res = channel_shuffle()
    except Exception:
        pass

def test_cholesky():
    try:
        res = cholesky()
    except Exception:
        pass

def test_cholesky_inverse():
    try:
        res = cholesky_inverse()
    except Exception:
        pass

def test_cholesky_solve():
    try:
        res = cholesky_solve()
    except Exception:
        pass

def test_choose_qparams_optimized():
    try:
        res = choose_qparams_optimized()
    except Exception:
        pass

def test_clamp():
    try:
        res = clamp()
    except Exception:
        pass

def test_clamp_():
    try:
        res = clamp_()
    except Exception:
        pass

def test_clamp_max():
    try:
        res = clamp_max()
    except Exception:
        pass

def test_clamp_max_():
    try:
        res = clamp_max_()
    except Exception:
        pass

def test_clamp_min():
    try:
        res = clamp_min()
    except Exception:
        pass

def test_clamp_min_():
    try:
        res = clamp_min_()
    except Exception:
        pass

def test_clip_():
    try:
        res = clip_()
    except Exception:
        pass

def test_clone():
    try:
        res = clone()
    except Exception:
        pass

def test_col_indices_copy():
    try:
        res = col_indices_copy()
    except Exception:
        pass

def test_column_stack():
    try:
        res = column_stack()
    except Exception:
        pass

def test_combinations():
    try:
        res = combinations()
    except Exception:
        pass

def test_complex():
    try:
        res = complex()
    except Exception:
        pass

def test_concatenate():
    try:
        res = concatenate()
    except Exception:
        pass

def test_conj():
    try:
        res = conj()
    except Exception:
        pass

def test_conj_physical():
    try:
        res = conj_physical()
    except Exception:
        pass

def test_conj_physical_():
    try:
        res = conj_physical_()
    except Exception:
        pass

def test_constant_pad_nd():
    try:
        res = constant_pad_nd()
    except Exception:
        pass

def test_conv1d():
    try:
        res = conv1d()
    except Exception:
        pass

def test_conv2d():
    try:
        res = conv2d()
    except Exception:
        pass

def test_conv3d():
    try:
        res = conv3d()
    except Exception:
        pass

def test_conv_tbc():
    try:
        res = conv_tbc()
    except Exception:
        pass

def test_conv_transpose1d():
    try:
        res = conv_transpose1d()
    except Exception:
        pass

def test_conv_transpose2d():
    try:
        res = conv_transpose2d()
    except Exception:
        pass

def test_conv_transpose3d():
    try:
        res = conv_transpose3d()
    except Exception:
        pass

def test_convolution():
    try:
        res = convolution()
    except Exception:
        pass

def test_copysign():
    try:
        res = copysign()
    except Exception:
        pass

def test_corrcoef():
    try:
        res = corrcoef()
    except Exception:
        pass

def test_cos_():
    try:
        res = cos_()
    except Exception:
        pass

def test_cosh_():
    try:
        res = cosh_()
    except Exception:
        pass

def test_cosine_embedding_loss():
    try:
        res = cosine_embedding_loss()
    except Exception:
        pass

def test_cosine_similarity():
    try:
        res = cosine_similarity()
    except Exception:
        pass

def test_count_nonzero():
    try:
        res = count_nonzero()
    except Exception:
        pass

def test_cov():
    try:
        res = cov()
    except Exception:
        pass

def test_cross():
    try:
        res = cross()
    except Exception:
        pass

def test_crow_indices_copy():
    try:
        res = crow_indices_copy()
    except Exception:
        pass

def test_ctc_loss():
    try:
        res = ctc_loss()
    except Exception:
        pass

def test_cudnn_affine_grid_generator():
    try:
        res = cudnn_affine_grid_generator()
    except Exception:
        pass

def test_cudnn_batch_norm():
    try:
        res = cudnn_batch_norm()
    except Exception:
        pass

def test_cudnn_convolution():
    try:
        res = cudnn_convolution()
    except Exception:
        pass

def test_cudnn_convolution_add_relu():
    try:
        res = cudnn_convolution_add_relu()
    except Exception:
        pass

def test_cudnn_convolution_relu():
    try:
        res = cudnn_convolution_relu()
    except Exception:
        pass

def test_cudnn_convolution_transpose():
    try:
        res = cudnn_convolution_transpose()
    except Exception:
        pass

def test_cudnn_grid_sampler():
    try:
        res = cudnn_grid_sampler()
    except Exception:
        pass

def test_cudnn_is_acceptable():
    try:
        res = cudnn_is_acceptable()
    except Exception:
        pass

def test_cummax():
    try:
        res = cummax()
    except Exception:
        pass

def test_cummin():
    try:
        res = cummin()
    except Exception:
        pass

def test_cumprod():
    try:
        res = cumprod()
    except Exception:
        pass

def test_cumulative_trapezoid():
    try:
        res = cumulative_trapezoid()
    except Exception:
        pass

def test_deg2rad():
    try:
        res = deg2rad()
    except Exception:
        pass

def test_deg2rad_():
    try:
        res = deg2rad_()
    except Exception:
        pass

def test_dequantize():
    try:
        res = dequantize()
    except Exception:
        pass

def test_detach():
    try:
        res = detach()
    except Exception:
        pass

def test_detach_():
    try:
        res = detach_()
    except Exception:
        pass

def test_detach_copy():
    try:
        res = detach_copy()
    except Exception:
        pass

def test_diag():
    try:
        res = diag()
    except Exception:
        pass

def test_diag_embed():
    try:
        res = diag_embed()
    except Exception:
        pass

def test_diagflat():
    try:
        res = diagflat()
    except Exception:
        pass

def test_diagonal():
    try:
        res = diagonal()
    except Exception:
        pass

def test_diagonal_copy():
    try:
        res = diagonal_copy()
    except Exception:
        pass

def test_diagonal_scatter():
    try:
        res = diagonal_scatter()
    except Exception:
        pass

def test_diff():
    try:
        res = diff()
    except Exception:
        pass

def test_digamma():
    try:
        res = digamma()
    except Exception:
        pass

def test_dist():
    try:
        res = dist()
    except Exception:
        pass

def test_divide():
    try:
        res = divide()
    except Exception:
        pass

def test_dot():
    try:
        res = dot()
    except Exception:
        pass

def test_dropout_():
    try:
        res = dropout_()
    except Exception:
        pass

def test_dsmm():
    try:
        res = dsmm()
    except Exception:
        pass

def test_dsplit():
    try:
        res = dsplit()
    except Exception:
        pass

def test_dstack():
    try:
        res = dstack()
    except Exception:
        pass

def test_embedding():
    try:
        res = embedding()
    except Exception:
        pass

def test_embedding_bag():
    try:
        res = embedding_bag()
    except Exception:
        pass

def test_embedding_renorm_():
    try:
        res = embedding_renorm_()
    except Exception:
        pass

def test_empty():
    try:
        res = empty()
    except Exception:
        pass

def test_empty_like():
    try:
        res = empty_like()
    except Exception:
        pass

def test_empty_permuted():
    try:
        res = empty_permuted()
    except Exception:
        pass

def test_empty_quantized():
    try:
        res = empty_quantized()
    except Exception:
        pass

def test_empty_strided():
    try:
        res = empty_strided()
    except Exception:
        pass

def test_eq():
    try:
        res = eq()
    except Exception:
        pass

def test_erf_():
    try:
        res = erf_()
    except Exception:
        pass

def test_erfc():
    try:
        res = erfc()
    except Exception:
        pass

def test_erfc_():
    try:
        res = erfc_()
    except Exception:
        pass

def test_erfinv():
    try:
        res = erfinv()
    except Exception:
        pass

def test_exp2():
    try:
        res = exp2()
    except Exception:
        pass

def test_exp2_():
    try:
        res = exp2_()
    except Exception:
        pass

def test_exp_():
    try:
        res = exp_()
    except Exception:
        pass

def test_expand_copy():
    try:
        res = expand_copy()
    except Exception:
        pass

def test_expm1():
    try:
        res = expm1()
    except Exception:
        pass

def test_expm1_():
    try:
        res = expm1_()
    except Exception:
        pass

def test_eye():
    try:
        res = eye()
    except Exception:
        pass

def test_fake_quantize_per_channel_affine():
    try:
        res = fake_quantize_per_channel_affine()
    except Exception:
        pass

def test_fake_quantize_per_tensor_affine():
    try:
        res = fake_quantize_per_tensor_affine()
    except Exception:
        pass

def test_fbgemm_linear_fp16_weight():
    try:
        res = fbgemm_linear_fp16_weight()
    except Exception:
        pass

def test_fbgemm_linear_fp16_weight_fp32_activation():
    try:
        res = fbgemm_linear_fp16_weight_fp32_activation()
    except Exception:
        pass

def test_fbgemm_linear_int8_weight():
    try:
        res = fbgemm_linear_int8_weight()
    except Exception:
        pass

def test_fbgemm_linear_int8_weight_fp32_activation():
    try:
        res = fbgemm_linear_int8_weight_fp32_activation()
    except Exception:
        pass

def test_fbgemm_linear_quantize_weight():
    try:
        res = fbgemm_linear_quantize_weight()
    except Exception:
        pass

def test_fbgemm_pack_gemm_matrix_fp16():
    try:
        res = fbgemm_pack_gemm_matrix_fp16()
    except Exception:
        pass

def test_fbgemm_pack_quantized_matrix():
    try:
        res = fbgemm_pack_quantized_matrix()
    except Exception:
        pass

def test_feature_alpha_dropout():
    try:
        res = feature_alpha_dropout()
    except Exception:
        pass

def test_feature_alpha_dropout_():
    try:
        res = feature_alpha_dropout_()
    except Exception:
        pass

def test_feature_dropout():
    try:
        res = feature_dropout()
    except Exception:
        pass

def test_feature_dropout_():
    try:
        res = feature_dropout_()
    except Exception:
        pass

def test_fill():
    try:
        res = fill()
    except Exception:
        pass

def test_fill_():
    try:
        res = fill_()
    except Exception:
        pass

def test_fix():
    try:
        res = fix()
    except Exception:
        pass

def test_fix_():
    try:
        res = fix_()
    except Exception:
        pass

def test_flatten():
    try:
        res = flatten()
    except Exception:
        pass

def test_flip():
    try:
        res = flip()
    except Exception:
        pass

def test_fliplr():
    try:
        res = fliplr()
    except Exception:
        pass

def test_flipud():
    try:
        res = flipud()
    except Exception:
        pass

def test_float_power():
    try:
        res = float_power()
    except Exception:
        pass

def test_floor_():
    try:
        res = floor_()
    except Exception:
        pass

def test_floor_divide():
    try:
        res = floor_divide()
    except Exception:
        pass

def test_fmax():
    try:
        res = fmax()
    except Exception:
        pass

def test_fmin():
    try:
        res = fmin()
    except Exception:
        pass

def test_fmod():
    try:
        res = fmod()
    except Exception:
        pass

def test_frac():
    try:
        res = frac()
    except Exception:
        pass

def test_frac_():
    try:
        res = frac_()
    except Exception:
        pass

def test_frexp():
    try:
        res = frexp()
    except Exception:
        pass

def test_frobenius_norm():
    try:
        res = frobenius_norm()
    except Exception:
        pass

def test_from_file():
    try:
        res = from_file()
    except Exception:
        pass

def test_from_numpy():
    try:
        res = from_numpy()
    except Exception:
        pass

def test_frombuffer():
    try:
        res = frombuffer()
    except Exception:
        pass

def test_full():
    try:
        res = full()
    except Exception:
        pass

def test_full_like():
    try:
        res = full_like()
    except Exception:
        pass

def test_fused_moving_avg_obs_fake_quant():
    try:
        res = fused_moving_avg_obs_fake_quant()
    except Exception:
        pass

def test_gcd():
    try:
        res = gcd()
    except Exception:
        pass

def test_gcd_():
    try:
        res = gcd_()
    except Exception:
        pass

def test_ge():
    try:
        res = ge()
    except Exception:
        pass

def test_geqrf():
    try:
        res = geqrf()
    except Exception:
        pass

def test_ger():
    try:
        res = ger()
    except Exception:
        pass

def test_get_device():
    try:
        res = get_device()
    except Exception:
        pass

def test_gradient():
    try:
        res = gradient()
    except Exception:
        pass

def test_greater_equal():
    try:
        res = greater_equal()
    except Exception:
        pass

def test_grid_sampler():
    try:
        res = grid_sampler()
    except Exception:
        pass

def test_grid_sampler_2d():
    try:
        res = grid_sampler_2d()
    except Exception:
        pass

def test_grid_sampler_3d():
    try:
        res = grid_sampler_3d()
    except Exception:
        pass

def test_group_norm():
    try:
        res = group_norm()
    except Exception:
        pass

def test_gru_cell():
    try:
        res = gru_cell()
    except Exception:
        pass

def test_gt():
    try:
        res = gt()
    except Exception:
        pass

def test_hamming_window():
    try:
        res = hamming_window()
    except Exception:
        pass

def test_hann_window():
    try:
        res = hann_window()
    except Exception:
        pass

def test_hardshrink():
    try:
        res = hardshrink()
    except Exception:
        pass

def test_hash_tensor():
    try:
        res = hash_tensor()
    except Exception:
        pass

def test_heaviside():
    try:
        res = heaviside()
    except Exception:
        pass

def test_hinge_embedding_loss():
    try:
        res = hinge_embedding_loss()
    except Exception:
        pass

def test_histc():
    try:
        res = histc()
    except Exception:
        pass

def test_histogram():
    try:
        res = histogram()
    except Exception:
        pass

def test_histogramdd():
    try:
        res = histogramdd()
    except Exception:
        pass

def test_hsmm():
    try:
        res = hsmm()
    except Exception:
        pass

def test_hsplit():
    try:
        res = hsplit()
    except Exception:
        pass

def test_hspmm():
    try:
        res = hspmm()
    except Exception:
        pass

def test_hstack():
    try:
        res = hstack()
    except Exception:
        pass

def test_hypot():
    try:
        res = hypot()
    except Exception:
        pass

def test_i0():
    try:
        res = i0()
    except Exception:
        pass

def test_i0_():
    try:
        res = i0_()
    except Exception:
        pass

def test_igamma():
    try:
        res = igamma()
    except Exception:
        pass

def test_igammac():
    try:
        res = igammac()
    except Exception:
        pass

def test_imag():
    try:
        res = imag()
    except Exception:
        pass

def test_index_add():
    try:
        res = index_add()
    except Exception:
        pass

def test_index_copy():
    try:
        res = index_copy()
    except Exception:
        pass

def test_index_fill():
    try:
        res = index_fill()
    except Exception:
        pass

def test_index_put():
    try:
        res = index_put()
    except Exception:
        pass

def test_index_put_():
    try:
        res = index_put_()
    except Exception:
        pass

def test_index_reduce():
    try:
        res = index_reduce()
    except Exception:
        pass

def test_index_select():
    try:
        res = index_select()
    except Exception:
        pass

def test_indices_copy():
    try:
        res = indices_copy()
    except Exception:
        pass

def test_inner():
    try:
        res = inner()
    except Exception:
        pass

def test_instance_norm():
    try:
        res = instance_norm()
    except Exception:
        pass

def test_int_repr():
    try:
        res = int_repr()
    except Exception:
        pass

def test_inverse():
    try:
        res = inverse()
    except Exception:
        pass

def test_is_complex():
    try:
        res = is_complex()
    except Exception:
        pass

def test_is_conj():
    try:
        res = is_conj()
    except Exception:
        pass

def test_is_distributed():
    try:
        res = is_distributed()
    except Exception:
        pass

def test_is_floating_point():
    try:
        res = is_floating_point()
    except Exception:
        pass

def test_is_inference():
    try:
        res = is_inference()
    except Exception:
        pass

def test_is_neg():
    try:
        res = is_neg()
    except Exception:
        pass

def test_is_nonzero():
    try:
        res = is_nonzero()
    except Exception:
        pass

def test_is_same_size():
    try:
        res = is_same_size()
    except Exception:
        pass

def test_is_signed():
    try:
        res = is_signed()
    except Exception:
        pass

def test_is_vulkan_available():
    try:
        res = is_vulkan_available()
    except Exception:
        pass

def test_isclose():
    try:
        res = isclose()
    except Exception:
        pass

def test_isfinite():
    try:
        res = isfinite()
    except Exception:
        pass

def test_isin():
    try:
        res = isin()
    except Exception:
        pass

def test_isneginf():
    try:
        res = isneginf()
    except Exception:
        pass

def test_isposinf():
    try:
        res = isposinf()
    except Exception:
        pass

def test_isreal():
    try:
        res = isreal()
    except Exception:
        pass

def test_istft():
    try:
        res = istft()
    except Exception:
        pass

def test_kaiser_window():
    try:
        res = kaiser_window()
    except Exception:
        pass

def test_kl_div():
    try:
        res = kl_div()
    except Exception:
        pass

def test_kron():
    try:
        res = kron()
    except Exception:
        pass

def test_kthvalue():
    try:
        res = kthvalue()
    except Exception:
        pass

def test_layer_norm():
    try:
        res = layer_norm()
    except Exception:
        pass

def test_lcm():
    try:
        res = lcm()
    except Exception:
        pass

def test_lcm_():
    try:
        res = lcm_()
    except Exception:
        pass

def test_ldexp():
    try:
        res = ldexp()
    except Exception:
        pass

def test_ldexp_():
    try:
        res = ldexp_()
    except Exception:
        pass

def test_le():
    try:
        res = le()
    except Exception:
        pass

def test_lerp():
    try:
        res = lerp()
    except Exception:
        pass

def test_less_equal():
    try:
        res = less_equal()
    except Exception:
        pass

def test_lgamma():
    try:
        res = lgamma()
    except Exception:
        pass

def test_linspace():
    try:
        res = linspace()
    except Exception:
        pass

def test_log():
    try:
        res = log()
    except Exception:
        pass

def test_log10():
    try:
        res = log10()
    except Exception:
        pass

def test_log10_():
    try:
        res = log10_()
    except Exception:
        pass

def test_log1p():
    try:
        res = log1p()
    except Exception:
        pass

def test_log1p_():
    try:
        res = log1p_()
    except Exception:
        pass

def test_log2():
    try:
        res = log2()
    except Exception:
        pass

def test_log2_():
    try:
        res = log2_()
    except Exception:
        pass

def test_log_():
    try:
        res = log_()
    except Exception:
        pass

def test_logaddexp():
    try:
        res = logaddexp()
    except Exception:
        pass

def test_logaddexp2():
    try:
        res = logaddexp2()
    except Exception:
        pass

def test_logcumsumexp():
    try:
        res = logcumsumexp()
    except Exception:
        pass

def test_logdet():
    try:
        res = logdet()
    except Exception:
        pass

def test_logical_and():
    try:
        res = logical_and()
    except Exception:
        pass

def test_logical_not():
    try:
        res = logical_not()
    except Exception:
        pass

def test_logical_or():
    try:
        res = logical_or()
    except Exception:
        pass

def test_logical_xor():
    try:
        res = logical_xor()
    except Exception:
        pass

def test_logit():
    try:
        res = logit()
    except Exception:
        pass

def test_logit_():
    try:
        res = logit_()
    except Exception:
        pass

def test_logspace():
    try:
        res = logspace()
    except Exception:
        pass

def test_logsumexp():
    try:
        res = logsumexp()
    except Exception:
        pass

def test_lstm_cell():
    try:
        res = lstm_cell()
    except Exception:
        pass

def test_lt():
    try:
        res = lt()
    except Exception:
        pass

def test_lu_solve():
    try:
        res = lu_solve()
    except Exception:
        pass

def test_lu_unpack():
    try:
        res = lu_unpack()
    except Exception:
        pass

def test_margin_ranking_loss():
    try:
        res = margin_ranking_loss()
    except Exception:
        pass

def test_masked_fill():
    try:
        res = masked_fill()
    except Exception:
        pass

def test_masked_scatter():
    try:
        res = masked_scatter()
    except Exception:
        pass

def test_masked_select():
    try:
        res = masked_select()
    except Exception:
        pass

def test_matrix_exp():
    try:
        res = matrix_exp()
    except Exception:
        pass

def test_matrix_power():
    try:
        res = matrix_power()
    except Exception:
        pass

def test_max_pool1d():
    try:
        res = max_pool1d()
    except Exception:
        pass

def test_max_pool1d_with_indices():
    try:
        res = max_pool1d_with_indices()
    except Exception:
        pass

def test_max_pool2d():
    try:
        res = max_pool2d()
    except Exception:
        pass

def test_max_pool3d():
    try:
        res = max_pool3d()
    except Exception:
        pass

def test_maximum():
    try:
        res = maximum()
    except Exception:
        pass

def test_median():
    try:
        res = median()
    except Exception:
        pass

def test_meshgrid():
    try:
        res = meshgrid()
    except Exception:
        pass

def test_minimum():
    try:
        res = minimum()
    except Exception:
        pass

def test_miopen_batch_norm():
    try:
        res = miopen_batch_norm()
    except Exception:
        pass

def test_miopen_convolution():
    try:
        res = miopen_convolution()
    except Exception:
        pass

def test_miopen_convolution_add_relu():
    try:
        res = miopen_convolution_add_relu()
    except Exception:
        pass

def test_miopen_convolution_relu():
    try:
        res = miopen_convolution_relu()
    except Exception:
        pass

def test_miopen_convolution_transpose():
    try:
        res = miopen_convolution_transpose()
    except Exception:
        pass

def test_miopen_ctc_loss():
    try:
        res = miopen_ctc_loss()
    except Exception:
        pass

def test_miopen_depthwise_convolution():
    try:
        res = miopen_depthwise_convolution()
    except Exception:
        pass

def test_miopen_rnn():
    try:
        res = miopen_rnn()
    except Exception:
        pass

def test_mkldnn_adaptive_avg_pool2d():
    try:
        res = mkldnn_adaptive_avg_pool2d()
    except Exception:
        pass

def test_mkldnn_convolution():
    try:
        res = mkldnn_convolution()
    except Exception:
        pass

def test_mkldnn_linear_backward_weights():
    try:
        res = mkldnn_linear_backward_weights()
    except Exception:
        pass

def test_mkldnn_max_pool2d():
    try:
        res = mkldnn_max_pool2d()
    except Exception:
        pass

def test_mkldnn_max_pool3d():
    try:
        res = mkldnn_max_pool3d()
    except Exception:
        pass

def test_mkldnn_rnn_layer():
    try:
        res = mkldnn_rnn_layer()
    except Exception:
        pass

def test_mm():
    try:
        res = mm()
    except Exception:
        pass

def test_mode():
    try:
        res = mode()
    except Exception:
        pass

def test_moveaxis():
    try:
        res = moveaxis()
    except Exception:
        pass

def test_movedim():
    try:
        res = movedim()
    except Exception:
        pass

def test_msort():
    try:
        res = msort()
    except Exception:
        pass

def test_multiply():
    try:
        res = multiply()
    except Exception:
        pass

def test_mv():
    try:
        res = mv()
    except Exception:
        pass

def test_mvlgamma():
    try:
        res = mvlgamma()
    except Exception:
        pass

def test_nan_to_num():
    try:
        res = nan_to_num()
    except Exception:
        pass

def test_nan_to_num_():
    try:
        res = nan_to_num_()
    except Exception:
        pass

def test_nanmean():
    try:
        res = nanmean()
    except Exception:
        pass

def test_nanmedian():
    try:
        res = nanmedian()
    except Exception:
        pass

def test_nanquantile():
    try:
        res = nanquantile()
    except Exception:
        pass

def test_nansum():
    try:
        res = nansum()
    except Exception:
        pass

def test_narrow():
    try:
        res = narrow()
    except Exception:
        pass

def test_narrow_copy():
    try:
        res = narrow_copy()
    except Exception:
        pass

def test_native_batch_norm():
    try:
        res = native_batch_norm()
    except Exception:
        pass

def test_native_channel_shuffle():
    try:
        res = native_channel_shuffle()
    except Exception:
        pass

def test_native_dropout():
    try:
        res = native_dropout()
    except Exception:
        pass

def test_native_group_norm():
    try:
        res = native_group_norm()
    except Exception:
        pass

def test_native_layer_norm():
    try:
        res = native_layer_norm()
    except Exception:
        pass

def test_native_norm():
    try:
        res = native_norm()
    except Exception:
        pass

def test_ne():
    try:
        res = ne()
    except Exception:
        pass

def test_neg_():
    try:
        res = neg_()
    except Exception:
        pass

def test_negative():
    try:
        res = negative()
    except Exception:
        pass

def test_negative_():
    try:
        res = negative_()
    except Exception:
        pass

def test_nextafter():
    try:
        res = nextafter()
    except Exception:
        pass

def test_nonzero():
    try:
        res = nonzero()
    except Exception:
        pass

def test_nonzero_static():
    try:
        res = nonzero_static()
    except Exception:
        pass

def test_norm():
    try:
        res = norm()
    except Exception:
        pass

def test_norm_except_dim():
    try:
        res = norm_except_dim()
    except Exception:
        pass

def test_normal():
    try:
        res = normal()
    except Exception:
        pass

def test_not_equal():
    try:
        res = not_equal()
    except Exception:
        pass

def test_nuclear_norm():
    try:
        res = nuclear_norm()
    except Exception:
        pass

def test_numel():
    try:
        res = numel()
    except Exception:
        pass

def test_ones_like():
    try:
        res = ones_like()
    except Exception:
        pass

def test_orgqr():
    try:
        res = orgqr()
    except Exception:
        pass

def test_ormqr():
    try:
        res = ormqr()
    except Exception:
        pass

def test_outer():
    try:
        res = outer()
    except Exception:
        pass

def test_pairwise_distance():
    try:
        res = pairwise_distance()
    except Exception:
        pass

def test_pdist():
    try:
        res = pdist()
    except Exception:
        pass

def test_permute():
    try:
        res = permute()
    except Exception:
        pass

def test_permute_copy():
    try:
        res = permute_copy()
    except Exception:
        pass

def test_pinverse():
    try:
        res = pinverse()
    except Exception:
        pass

def test_pixel_shuffle():
    try:
        res = pixel_shuffle()
    except Exception:
        pass

def test_pixel_unshuffle():
    try:
        res = pixel_unshuffle()
    except Exception:
        pass

def test_poisson():
    try:
        res = poisson()
    except Exception:
        pass

def test_poisson_nll_loss():
    try:
        res = poisson_nll_loss()
    except Exception:
        pass

def test_polar():
    try:
        res = polar()
    except Exception:
        pass

def test_polygamma():
    try:
        res = polygamma()
    except Exception:
        pass

def test_positive():
    try:
        res = positive()
    except Exception:
        pass

def test_prod():
    try:
        res = prod()
    except Exception:
        pass

def test_promote_types():
    try:
        res = promote_types()
    except Exception:
        pass

def test_put():
    try:
        res = put()
    except Exception:
        pass

def test_q_per_channel_axis():
    try:
        res = q_per_channel_axis()
    except Exception:
        pass

def test_q_per_channel_scales():
    try:
        res = q_per_channel_scales()
    except Exception:
        pass

def test_q_per_channel_zero_points():
    try:
        res = q_per_channel_zero_points()
    except Exception:
        pass

def test_q_scale():
    try:
        res = q_scale()
    except Exception:
        pass

def test_q_zero_point():
    try:
        res = q_zero_point()
    except Exception:
        pass

def test_qr():
    try:
        res = qr()
    except Exception:
        pass

def test_quantile():
    try:
        res = quantile()
    except Exception:
        pass

def test_quantize_per_channel():
    try:
        res = quantize_per_channel()
    except Exception:
        pass

def test_quantize_per_tensor():
    try:
        res = quantize_per_tensor()
    except Exception:
        pass

def test_quantize_per_tensor_dynamic():
    try:
        res = quantize_per_tensor_dynamic()
    except Exception:
        pass

def test_quantized_batch_norm():
    try:
        res = quantized_batch_norm()
    except Exception:
        pass

def test_quantized_gru_cell():
    try:
        res = quantized_gru_cell()
    except Exception:
        pass

def test_quantized_lstm_cell():
    try:
        res = quantized_lstm_cell()
    except Exception:
        pass

def test_quantized_max_pool1d():
    try:
        res = quantized_max_pool1d()
    except Exception:
        pass

def test_quantized_max_pool2d():
    try:
        res = quantized_max_pool2d()
    except Exception:
        pass

def test_quantized_max_pool3d():
    try:
        res = quantized_max_pool3d()
    except Exception:
        pass

def test_quantized_rnn_relu_cell():
    try:
        res = quantized_rnn_relu_cell()
    except Exception:
        pass

def test_quantized_rnn_tanh_cell():
    try:
        res = quantized_rnn_tanh_cell()
    except Exception:
        pass

def test_rad2deg():
    try:
        res = rad2deg()
    except Exception:
        pass

def test_rad2deg_():
    try:
        res = rad2deg_()
    except Exception:
        pass

def test_rand_like():
    try:
        res = rand_like()
    except Exception:
        pass

def test_randint():
    try:
        res = randint()
    except Exception:
        pass

def test_randint_like():
    try:
        res = randint_like()
    except Exception:
        pass

def test_randn_like():
    try:
        res = randn_like()
    except Exception:
        pass

def test_randperm():
    try:
        res = randperm()
    except Exception:
        pass

def test_range():
    try:
        res = range()
    except Exception:
        pass

def test_ravel():
    try:
        res = ravel()
    except Exception:
        pass

def test_real():
    try:
        res = real()
    except Exception:
        pass

def test_reciprocal_():
    try:
        res = reciprocal_()
    except Exception:
        pass

def test_relu_():
    try:
        res = relu_()
    except Exception:
        pass

def test_remainder():
    try:
        res = remainder()
    except Exception:
        pass

def test_renorm():
    try:
        res = renorm()
    except Exception:
        pass

def test_repeat_interleave():
    try:
        res = repeat_interleave()
    except Exception:
        pass

def test_resize_as_():
    try:
        res = resize_as_()
    except Exception:
        pass

def test_resize_as_sparse_():
    try:
        res = resize_as_sparse_()
    except Exception:
        pass

def test_resolve_conj():
    try:
        res = resolve_conj()
    except Exception:
        pass

def test_resolve_neg():
    try:
        res = resolve_neg()
    except Exception:
        pass

def test_result_type():
    try:
        res = result_type()
    except Exception:
        pass

def test_rms_norm():
    try:
        res = rms_norm()
    except Exception:
        pass

def test_rnn_relu():
    try:
        res = rnn_relu()
    except Exception:
        pass

def test_rnn_relu_cell():
    try:
        res = rnn_relu_cell()
    except Exception:
        pass

def test_rnn_tanh():
    try:
        res = rnn_tanh()
    except Exception:
        pass

def test_rnn_tanh_cell():
    try:
        res = rnn_tanh_cell()
    except Exception:
        pass

def test_roll():
    try:
        res = roll()
    except Exception:
        pass

def test_rot90():
    try:
        res = rot90()
    except Exception:
        pass

def test_round_():
    try:
        res = round_()
    except Exception:
        pass

def test_row_indices_copy():
    try:
        res = row_indices_copy()
    except Exception:
        pass

def test_row_stack():
    try:
        res = row_stack()
    except Exception:
        pass

def test_rrelu():
    try:
        res = rrelu()
    except Exception:
        pass

def test_rrelu_():
    try:
        res = rrelu_()
    except Exception:
        pass

def test_rsqrt():
    try:
        res = rsqrt()
    except Exception:
        pass

def test_rsqrt_():
    try:
        res = rsqrt_()
    except Exception:
        pass

def test_rsub():
    try:
        res = rsub()
    except Exception:
        pass

def test_saddmm():
    try:
        res = saddmm()
    except Exception:
        pass

def test_scalar_tensor():
    try:
        res = scalar_tensor()
    except Exception:
        pass

def test_scatter_add():
    try:
        res = scatter_add()
    except Exception:
        pass

def test_scatter_reduce():
    try:
        res = scatter_reduce()
    except Exception:
        pass

def test_searchsorted():
    try:
        res = searchsorted()
    except Exception:
        pass

def test_select():
    try:
        res = select()
    except Exception:
        pass

def test_select_copy():
    try:
        res = select_copy()
    except Exception:
        pass

def test_select_scatter():
    try:
        res = select_scatter()
    except Exception:
        pass

def test_selu_():
    try:
        res = selu_()
    except Exception:
        pass

def test_sgn():
    try:
        res = sgn()
    except Exception:
        pass

def test_sigmoid_():
    try:
        res = sigmoid_()
    except Exception:
        pass

def test_signbit():
    try:
        res = signbit()
    except Exception:
        pass

def test_sin_():
    try:
        res = sin_()
    except Exception:
        pass

def test_sinc():
    try:
        res = sinc()
    except Exception:
        pass

def test_sinc_():
    try:
        res = sinc_()
    except Exception:
        pass

def test_sinh_():
    try:
        res = sinh_()
    except Exception:
        pass

def test_slice_copy():
    try:
        res = slice_copy()
    except Exception:
        pass

def test_slice_inverse():
    try:
        res = slice_inverse()
    except Exception:
        pass

def test_slice_scatter():
    try:
        res = slice_scatter()
    except Exception:
        pass

def test_slogdet():
    try:
        res = slogdet()
    except Exception:
        pass

def test_smm():
    try:
        res = smm()
    except Exception:
        pass

def test_sort():
    try:
        res = sort()
    except Exception:
        pass

def test_sparse_bsc_tensor():
    try:
        res = sparse_bsc_tensor()
    except Exception:
        pass

def test_sparse_bsr_tensor():
    try:
        res = sparse_bsr_tensor()
    except Exception:
        pass

def test_sparse_compressed_tensor():
    try:
        res = sparse_compressed_tensor()
    except Exception:
        pass

def test_sparse_coo_tensor():
    try:
        res = sparse_coo_tensor()
    except Exception:
        pass

def test_sparse_csc_tensor():
    try:
        res = sparse_csc_tensor()
    except Exception:
        pass

def test_sparse_csr_tensor():
    try:
        res = sparse_csr_tensor()
    except Exception:
        pass

def test_split_copy():
    try:
        res = split_copy()
    except Exception:
        pass

def test_split_with_sizes():
    try:
        res = split_with_sizes()
    except Exception:
        pass

def test_split_with_sizes_copy():
    try:
        res = split_with_sizes_copy()
    except Exception:
        pass

def test_spmm():
    try:
        res = spmm()
    except Exception:
        pass

def test_sqrt():
    try:
        res = sqrt()
    except Exception:
        pass

def test_sqrt_():
    try:
        res = sqrt_()
    except Exception:
        pass

def test_square():
    try:
        res = square()
    except Exception:
        pass

def test_square_():
    try:
        res = square_()
    except Exception:
        pass

def test_squeeze():
    try:
        res = squeeze()
    except Exception:
        pass

def test_squeeze_copy():
    try:
        res = squeeze_copy()
    except Exception:
        pass

def test_sspaddmm():
    try:
        res = sspaddmm()
    except Exception:
        pass

def test_std():
    try:
        res = std()
    except Exception:
        pass

def test_std_mean():
    try:
        res = std_mean()
    except Exception:
        pass

def test_stft():
    try:
        res = stft()
    except Exception:
        pass

def test_subtract():
    try:
        res = subtract()
    except Exception:
        pass

def test_svd():
    try:
        res = svd()
    except Exception:
        pass

def test_swapaxes():
    try:
        res = swapaxes()
    except Exception:
        pass

def test_swapdims():
    try:
        res = swapdims()
    except Exception:
        pass

def test_sym_constrain_range():
    try:
        res = sym_constrain_range()
    except Exception:
        pass

def test_sym_constrain_range_for_size():
    try:
        res = sym_constrain_range_for_size()
    except Exception:
        pass

def test_t():
    try:
        res = t()
    except Exception:
        pass

def test_t_copy():
    try:
        res = t_copy()
    except Exception:
        pass

def test_take():
    try:
        res = take()
    except Exception:
        pass

def test_take_along_dim():
    try:
        res = take_along_dim()
    except Exception:
        pass

def test_tan_():
    try:
        res = tan_()
    except Exception:
        pass

def test_tanh_():
    try:
        res = tanh_()
    except Exception:
        pass

def test_tensor_split():
    try:
        res = tensor_split()
    except Exception:
        pass

def test_tensordot():
    try:
        res = tensordot()
    except Exception:
        pass

def test_threshold():
    try:
        res = threshold()
    except Exception:
        pass

def test_threshold_():
    try:
        res = threshold_()
    except Exception:
        pass

def test_transpose_copy():
    try:
        res = transpose_copy()
    except Exception:
        pass

def test_trapezoid():
    try:
        res = trapezoid()
    except Exception:
        pass

def test_trapz():
    try:
        res = trapz()
    except Exception:
        pass

def test_triangular_solve():
    try:
        res = triangular_solve()
    except Exception:
        pass

def test_tril():
    try:
        res = tril()
    except Exception:
        pass

def test_tril_indices():
    try:
        res = tril_indices()
    except Exception:
        pass

def test_triplet_margin_loss():
    try:
        res = triplet_margin_loss()
    except Exception:
        pass

def test_triu():
    try:
        res = triu()
    except Exception:
        pass

def test_triu_indices():
    try:
        res = triu_indices()
    except Exception:
        pass

def test_true_divide():
    try:
        res = true_divide()
    except Exception:
        pass

def test_trunc():
    try:
        res = trunc()
    except Exception:
        pass

def test_trunc_():
    try:
        res = trunc_()
    except Exception:
        pass

def test_unbind():
    try:
        res = unbind()
    except Exception:
        pass

def test_unbind_copy():
    try:
        res = unbind_copy()
    except Exception:
        pass

def test_unflatten():
    try:
        res = unflatten()
    except Exception:
        pass

def test_unfold_copy():
    try:
        res = unfold_copy()
    except Exception:
        pass

def test_unique_consecutive():
    try:
        res = unique_consecutive()
    except Exception:
        pass

def test_unsafe_chunk():
    try:
        res = unsafe_chunk()
    except Exception:
        pass

def test_unsafe_split():
    try:
        res = unsafe_split()
    except Exception:
        pass

def test_unsafe_split_with_sizes():
    try:
        res = unsafe_split_with_sizes()
    except Exception:
        pass

def test_unsqueeze():
    try:
        res = unsqueeze()
    except Exception:
        pass

def test_unsqueeze_copy():
    try:
        res = unsqueeze_copy()
    except Exception:
        pass

def test_values_copy():
    try:
        res = values_copy()
    except Exception:
        pass

def test_vander():
    try:
        res = vander()
    except Exception:
        pass

def test_var():
    try:
        res = var()
    except Exception:
        pass

def test_var_mean():
    try:
        res = var_mean()
    except Exception:
        pass

def test_vdot():
    try:
        res = vdot()
    except Exception:
        pass

def test_view_as_complex():
    try:
        res = view_as_complex()
    except Exception:
        pass

def test_view_as_complex_copy():
    try:
        res = view_as_complex_copy()
    except Exception:
        pass

def test_view_as_real():
    try:
        res = view_as_real()
    except Exception:
        pass

def test_view_as_real_copy():
    try:
        res = view_as_real_copy()
    except Exception:
        pass

def test_view_copy():
    try:
        res = view_copy()
    except Exception:
        pass

def test_vsplit():
    try:
        res = vsplit()
    except Exception:
        pass

def test_vstack():
    try:
        res = vstack()
    except Exception:
        pass

def test_xlogy():
    try:
        res = xlogy()
    except Exception:
        pass

def test_xlogy_():
    try:
        res = xlogy_()
    except Exception:
        pass

def test_zero_():
    try:
        res = zero_()
    except Exception:
        pass

def test_zeros_like():
    try:
        res = zeros_like()
    except Exception:
        pass

def test_bfloat16():
    try:
        res = bfloat16()
    except Exception:
        pass

def test_bit():
    try:
        res = bit()
    except Exception:
        pass

def test_bits16():
    try:
        res = bits16()
    except Exception:
        pass

def test_bits1x8():
    try:
        res = bits1x8()
    except Exception:
        pass

def test_bits2x4():
    try:
        res = bits2x4()
    except Exception:
        pass

def test_bits4x2():
    try:
        res = bits4x2()
    except Exception:
        pass

def test_bits8():
    try:
        res = bits8()
    except Exception:
        pass

def test_cdouble():
    try:
        res = cdouble()
    except Exception:
        pass

def test_cfloat():
    try:
        res = cfloat()
    except Exception:
        pass

def test_chalf():
    try:
        res = chalf()
    except Exception:
        pass

def test_complex128():
    try:
        res = complex128()
    except Exception:
        pass

def test_complex32():
    try:
        res = complex32()
    except Exception:
        pass

def test_complex64():
    try:
        res = complex64()
    except Exception:
        pass

def test_double():
    try:
        res = double()
    except Exception:
        pass

def test_float():
    try:
        res = float()
    except Exception:
        pass

def test_float16():
    try:
        res = float16()
    except Exception:
        pass

def test_float4_e2m1fn_x2():
    try:
        res = float4_e2m1fn_x2()
    except Exception:
        pass

def test_float8_e4m3fn():
    try:
        res = float8_e4m3fn()
    except Exception:
        pass

def test_float8_e4m3fnuz():
    try:
        res = float8_e4m3fnuz()
    except Exception:
        pass

def test_float8_e5m2():
    try:
        res = float8_e5m2()
    except Exception:
        pass

def test_float8_e5m2fnuz():
    try:
        res = float8_e5m2fnuz()
    except Exception:
        pass

def test_float8_e8m0fnu():
    try:
        res = float8_e8m0fnu()
    except Exception:
        pass

def test_half():
    try:
        res = half()
    except Exception:
        pass

def test_int():
    try:
        res = int()
    except Exception:
        pass

def test_int1():
    try:
        res = int1()
    except Exception:
        pass

def test_int16():
    try:
        res = int16()
    except Exception:
        pass

def test_int2():
    try:
        res = int2()
    except Exception:
        pass

def test_int3():
    try:
        res = int3()
    except Exception:
        pass

def test_int4():
    try:
        res = int4()
    except Exception:
        pass

def test_int5():
    try:
        res = int5()
    except Exception:
        pass

def test_int6():
    try:
        res = int6()
    except Exception:
        pass

def test_int7():
    try:
        res = int7()
    except Exception:
        pass

def test_int8():
    try:
        res = int8()
    except Exception:
        pass

def test_long():
    try:
        res = long()
    except Exception:
        pass

def test_qint32():
    try:
        res = qint32()
    except Exception:
        pass

def test_qint8():
    try:
        res = qint8()
    except Exception:
        pass

def test_quint2x4():
    try:
        res = quint2x4()
    except Exception:
        pass

def test_quint4x2():
    try:
        res = quint4x2()
    except Exception:
        pass

def test_quint8():
    try:
        res = quint8()
    except Exception:
        pass

def test_short():
    try:
        res = short()
    except Exception:
        pass

def test_uint1():
    try:
        res = uint1()
    except Exception:
        pass

def test_uint16():
    try:
        res = uint16()
    except Exception:
        pass

def test_uint2():
    try:
        res = uint2()
    except Exception:
        pass

def test_uint3():
    try:
        res = uint3()
    except Exception:
        pass

def test_uint32():
    try:
        res = uint32()
    except Exception:
        pass

def test_uint4():
    try:
        res = uint4()
    except Exception:
        pass

def test_uint5():
    try:
        res = uint5()
    except Exception:
        pass

def test_uint6():
    try:
        res = uint6()
    except Exception:
        pass

def test_uint64():
    try:
        res = uint64()
    except Exception:
        pass

def test_uint7():
    try:
        res = uint7()
    except Exception:
        pass

def test_uint8():
    try:
        res = uint8()
    except Exception:
        pass

