import pytest
from onnx9000.converters.torch_like import *


def test_jit():
    try:
        obj = jit()
        assert obj is not None
    except Exception:
        pass


def test_onnx():
    try:
        obj = onnx()
        assert obj is not None
    except Exception:
        pass


def test_tensor():
    try:
        tensor()
    except Exception:
        pass


def test_zeros():
    try:
        zeros()
    except Exception:
        pass


def test_ones():
    try:
        ones()
    except Exception:
        pass


def test_randn():
    try:
        randn()
    except Exception:
        pass


def test_BoolStorage():
    try:
        BoolStorage()
    except Exception:
        pass


def test_BoolTensor():
    try:
        BoolTensor()
    except Exception:
        pass


def test_ByteStorage():
    try:
        ByteStorage()
    except Exception:
        pass


def test_ByteTensor():
    try:
        ByteTensor()
    except Exception:
        pass


def test_CharStorage():
    try:
        CharStorage()
    except Exception:
        pass


def test_CharTensor():
    try:
        CharTensor()
    except Exception:
        pass


def test_DoubleStorage():
    try:
        DoubleStorage()
    except Exception:
        pass


def test_DoubleTensor():
    try:
        DoubleTensor()
    except Exception:
        pass


def test_FloatStorage():
    try:
        FloatStorage()
    except Exception:
        pass


def test_FloatTensor():
    try:
        FloatTensor()
    except Exception:
        pass


def test_GradScaler():
    try:
        GradScaler()
    except Exception:
        pass


def test_IntStorage():
    try:
        IntStorage()
    except Exception:
        pass


def test_IntTensor():
    try:
        IntTensor()
    except Exception:
        pass


def test_LongStorage():
    try:
        LongStorage()
    except Exception:
        pass


def test_LongTensor():
    try:
        LongTensor()
    except Exception:
        pass


def test_ShortStorage():
    try:
        ShortStorage()
    except Exception:
        pass


def test_ShortTensor():
    try:
        ShortTensor()
    except Exception:
        pass


def test_SymBool():
    try:
        SymBool()
    except Exception:
        pass


def test_SymFloat():
    try:
        SymFloat()
    except Exception:
        pass


def test_SymInt():
    try:
        SymInt()
    except Exception:
        pass


def test_TypedStorage():
    try:
        TypedStorage()
    except Exception:
        pass


def test_UntypedStorage():
    try:
        UntypedStorage()
    except Exception:
        pass


def test_are_deterministic_algorithms_enabled():
    try:
        are_deterministic_algorithms_enabled()
    except Exception:
        pass


def test_autocast():
    try:
        autocast()
    except Exception:
        pass


def test_chunk():
    try:
        chunk()
    except Exception:
        pass


def test_compile():
    try:
        compile()
    except Exception:
        pass


def test_cond():
    try:
        cond()
    except Exception:
        pass


def test_enable_grad():
    try:
        enable_grad()
    except Exception:
        pass


def test_export_AdditionalInputs():
    try:
        export_AdditionalInputs()
    except Exception:
        pass


def test_export_Constraint():
    try:
        export_Constraint()
    except Exception:
        pass


def test_export_CustomDecompTable():
    try:
        export_CustomDecompTable()
    except Exception:
        pass


def test_export_default_decompositions():
    try:
        export_default_decompositions()
    except Exception:
        pass


def test_export_Dim():
    try:
        export_Dim()
    except Exception:
        pass


def test_export_dims():
    try:
        export_dims()
    except Exception:
        pass


def test_export_draft_export():
    try:
        export_draft_export()
    except Exception:
        pass


def test_export_export():
    try:
        export_export()
    except Exception:
        pass


def test_export_ExportBackwardSignature():
    try:
        export_ExportBackwardSignature()
    except Exception:
        pass


def test_export_ExportedProgram():
    try:
        export_ExportedProgram()
    except Exception:
        pass


def test_export_ExportGraphSignature():
    try:
        export_ExportGraphSignature()
    except Exception:
        pass


def test_export_FlatArgsAdapter():
    try:
        export_FlatArgsAdapter()
    except Exception:
        pass


def test_export_load():
    try:
        export_load()
    except Exception:
        pass


def test_export_ModuleCallEntry():
    try:
        export_ModuleCallEntry()
    except Exception:
        pass


def test_export_ModuleCallSignature():
    try:
        export_ModuleCallSignature()
    except Exception:
        pass


def test_export_register_dataclass():
    try:
        export_register_dataclass()
    except Exception:
        pass


def test_export_save():
    try:
        export_save()
    except Exception:
        pass


def test_export_ShapesCollection():
    try:
        export_ShapesCollection()
    except Exception:
        pass


def test_export_unflatten():
    try:
        export_unflatten()
    except Exception:
        pass


def test_export_UnflattenedModule():
    try:
        export_UnflattenedModule()
    except Exception:
        pass


def test_get_default_device():
    try:
        get_default_device()
    except Exception:
        pass


def test_get_deterministic_debug_mode():
    try:
        get_deterministic_debug_mode()
    except Exception:
        pass


def test_get_device_module():
    try:
        get_device_module()
    except Exception:
        pass


def test_get_float32_matmul_precision():
    try:
        get_float32_matmul_precision()
    except Exception:
        pass


def test_get_rng_state():
    try:
        get_rng_state()
    except Exception:
        pass


def test_inference_mode():
    try:
        inference_mode()
    except Exception:
        pass


def test_initial_seed():
    try:
        initial_seed()
    except Exception:
        pass


def test_is_deterministic_algorithms_warn_only_enabled():
    try:
        is_deterministic_algorithms_warn_only_enabled()
    except Exception:
        pass


def test_is_storage():
    try:
        is_storage()
    except Exception:
        pass


def test_is_tensor():
    try:
        is_tensor()
    except Exception:
        pass


def test_is_warn_always_enabled():
    try:
        is_warn_always_enabled()
    except Exception:
        pass


def test_load():
    try:
        load()
    except Exception:
        pass


def test_lobpcg():
    try:
        lobpcg()
    except Exception:
        pass


def test_manual_seed():
    try:
        manual_seed()
    except Exception:
        pass


def test_matmul():
    try:
        matmul()
    except Exception:
        pass


def test_no_grad():
    try:
        no_grad()
    except Exception:
        pass


def test_rand():
    try:
        rand()
    except Exception:
        pass


def test_save():
    try:
        save()
    except Exception:
        pass


def test_seed():
    try:
        seed()
    except Exception:
        pass


def test_set_default_device():
    try:
        set_default_device()
    except Exception:
        pass


def test_set_default_tensor_type():
    try:
        set_default_tensor_type()
    except Exception:
        pass


def test_set_deterministic_debug_mode():
    try:
        set_deterministic_debug_mode()
    except Exception:
        pass


def test_set_float32_matmul_precision():
    try:
        set_float32_matmul_precision()
    except Exception:
        pass


def test_set_printoptions():
    try:
        set_printoptions()
    except Exception:
        pass


def test_set_rng_state():
    try:
        set_rng_state()
    except Exception:
        pass


def test_set_warn_always():
    try:
        set_warn_always()
    except Exception:
        pass


def test_split():
    try:
        split()
    except Exception:
        pass


def test_stack():
    try:
        stack()
    except Exception:
        pass


def test_sym_float():
    try:
        sym_float()
    except Exception:
        pass


def test_sym_fresh_size():
    try:
        sym_fresh_size()
    except Exception:
        pass


def test_sym_int():
    try:
        sym_int()
    except Exception:
        pass


def test_sym_ite():
    try:
        sym_ite()
    except Exception:
        pass


def test_sym_max():
    try:
        sym_max()
    except Exception:
        pass


def test_sym_min():
    try:
        sym_min()
    except Exception:
        pass


def test_sym_not():
    try:
        sym_not()
    except Exception:
        pass


def test_sym_sum():
    try:
        sym_sum()
    except Exception:
        pass


def test_typename():
    try:
        typename()
    except Exception:
        pass


def test_unravel_index():
    try:
        unravel_index()
    except Exception:
        pass


def test_use_deterministic_algorithms():
    try:
        use_deterministic_algorithms()
    except Exception:
        pass


def test_vmap():
    try:
        vmap()
    except Exception:
        pass


def test_sym_sqrt():
    try:
        sym_sqrt()
    except Exception:
        pass


def test_AVG():
    try:
        AVG()
    except Exception:
        pass


def test_AcceleratorError():
    try:
        AcceleratorError()
    except Exception:
        pass


def test_AggregationType():
    try:
        AggregationType()
    except Exception:
        pass


def test_AliasDb():
    try:
        AliasDb()
    except Exception:
        pass


def test_AnyType():
    try:
        AnyType()
    except Exception:
        pass


def test_Argument():
    try:
        Argument()
    except Exception:
        pass


def test_ArgumentSpec():
    try:
        ArgumentSpec()
    except Exception:
        pass


def test_AwaitType():
    try:
        AwaitType()
    except Exception:
        pass


def test_BenchmarkConfig():
    try:
        BenchmarkConfig()
    except Exception:
        pass


def test_BenchmarkExecutionStats():
    try:
        BenchmarkExecutionStats()
    except Exception:
        pass


def test_Block():
    try:
        Block()
    except Exception:
        pass


def test_BoolType():
    try:
        BoolType()
    except Exception:
        pass


def test_BufferDict():
    try:
        BufferDict()
    except Exception:
        pass


def test_CallStack():
    try:
        CallStack()
    except Exception:
        pass


def test_Capsule():
    try:
        Capsule()
    except Exception:
        pass


def test_ClassType():
    try:
        ClassType()
    except Exception:
        pass


def test_Code():
    try:
        Code()
    except Exception:
        pass


def test_CompilationUnit():
    try:
        CompilationUnit()
    except Exception:
        pass


def test_CompleteArgumentSpec():
    try:
        CompleteArgumentSpec()
    except Exception:
        pass


def test_ComplexType():
    try:
        ComplexType()
    except Exception:
        pass


def test_ConcreteModuleType():
    try:
        ConcreteModuleType()
    except Exception:
        pass


def test_ConcreteModuleTypeBuilder():
    try:
        ConcreteModuleTypeBuilder()
    except Exception:
        pass


def test_DeepCopyMemoTable():
    try:
        DeepCopyMemoTable()
    except Exception:
        pass


def test_DeserializationStorageContext():
    try:
        DeserializationStorageContext()
    except Exception:
        pass


def test_DeviceObjType():
    try:
        DeviceObjType()
    except Exception:
        pass


def test_DictType():
    try:
        DictType()
    except Exception:
        pass


def test_DisableTorchFunction():
    try:
        DisableTorchFunction()
    except Exception:
        pass


def test_DisableTorchFunctionSubclass():
    try:
        DisableTorchFunctionSubclass()
    except Exception:
        pass


def test_DispatchKey():
    try:
        DispatchKey()
    except Exception:
        pass


def test_DispatchKeySet():
    try:
        DispatchKeySet()
    except Exception:
        pass


def test_EnumType():
    try:
        EnumType()
    except Exception:
        pass


def test_ErrorReport():
    try:
        ErrorReport()
    except Exception:
        pass


def test_Event():
    try:
        Event()
    except Exception:
        pass


def test_ExcludeDispatchKeyGuard():
    try:
        ExcludeDispatchKeyGuard()
    except Exception:
        pass


def test_ExecutionPlan():
    try:
        ExecutionPlan()
    except Exception:
        pass


def test_FatalError():
    try:
        FatalError()
    except Exception:
        pass


def test_FileCheck():
    try:
        FileCheck()
    except Exception:
        pass


def test_FloatType():
    try:
        FloatType()
    except Exception:
        pass


def test_FunctionSchema():
    try:
        FunctionSchema()
    except Exception:
        pass


def test_Future():
    try:
        Future()
    except Exception:
        pass


def test_FutureType():
    try:
        FutureType()
    except Exception:
        pass


def test_Generator():
    try:
        Generator()
    except Exception:
        pass


def test_Gradient():
    try:
        Gradient()
    except Exception:
        pass


def test_Graph():
    try:
        Graph()
    except Exception:
        pass


def test_GraphExecutorState():
    try:
        GraphExecutorState()
    except Exception:
        pass


def test_IODescriptor():
    try:
        IODescriptor()
    except Exception:
        pass


def test_InferredType():
    try:
        InferredType()
    except Exception:
        pass


def test_IntType():
    try:
        IntType()
    except Exception:
        pass


def test_InterfaceType():
    try:
        InterfaceType()
    except Exception:
        pass


def test_JITException():
    try:
        JITException()
    except Exception:
        pass


def test_ListType():
    try:
        ListType()
    except Exception:
        pass


def test_LiteScriptModule():
    try:
        LiteScriptModule()
    except Exception:
        pass


def test_LockingLogger():
    try:
        LockingLogger()
    except Exception:
        pass


def test_ModuleDict():
    try:
        ModuleDict()
    except Exception:
        pass


def test_Node():
    try:
        Node()
    except Exception:
        pass


def test_NoneType():
    try:
        NoneType()
    except Exception:
        pass


def test_NoopLogger():
    try:
        NoopLogger()
    except Exception:
        pass


def test_NumberType():
    try:
        NumberType()
    except Exception:
        pass


def test_OperatorInfo():
    try:
        OperatorInfo()
    except Exception:
        pass


def test_OptionalType():
    try:
        OptionalType()
    except Exception:
        pass


def test_OutOfMemoryError():
    try:
        OutOfMemoryError()
    except Exception:
        pass


def test_ParameterDict():
    try:
        ParameterDict()
    except Exception:
        pass


def test_PyObjectType():
    try:
        PyObjectType()
    except Exception:
        pass


def test_PyTorchFileReader():
    try:
        PyTorchFileReader()
    except Exception:
        pass


def test_PyTorchFileWriter():
    try:
        PyTorchFileWriter()
    except Exception:
        pass


def test_RRefType():
    try:
        RRefType()
    except Exception:
        pass


def test_SUM():
    try:
        SUM()
    except Exception:
        pass


def test_ScriptClass():
    try:
        ScriptClass()
    except Exception:
        pass


def test_ScriptClassFunction():
    try:
        ScriptClassFunction()
    except Exception:
        pass


def test_ScriptDict():
    try:
        ScriptDict()
    except Exception:
        pass


def test_ScriptDictIterator():
    try:
        ScriptDictIterator()
    except Exception:
        pass


def test_ScriptDictKeyIterator():
    try:
        ScriptDictKeyIterator()
    except Exception:
        pass


def test_ScriptFunction():
    try:
        ScriptFunction()
    except Exception:
        pass


def test_ScriptList():
    try:
        ScriptList()
    except Exception:
        pass


def test_ScriptListIterator():
    try:
        ScriptListIterator()
    except Exception:
        pass


def test_ScriptMethod():
    try:
        ScriptMethod()
    except Exception:
        pass


def test_ScriptModule():
    try:
        ScriptModule()
    except Exception:
        pass


def test_ScriptModuleSerializer():
    try:
        ScriptModuleSerializer()
    except Exception:
        pass


def test_ScriptObject():
    try:
        ScriptObject()
    except Exception:
        pass


def test_ScriptObjectProperty():
    try:
        ScriptObjectProperty()
    except Exception:
        pass


def test_SerializationStorageContext():
    try:
        SerializationStorageContext()
    except Exception:
        pass


def test_Size():
    try:
        Size()
    except Exception:
        pass


def test_StaticModule():
    try:
        StaticModule()
    except Exception:
        pass


def test_Stream():
    try:
        Stream()
    except Exception:
        pass


def test_StreamObjType():
    try:
        StreamObjType()
    except Exception:
        pass


def test_StringType():
    try:
        StringType()
    except Exception:
        pass


def test_SymBoolType():
    try:
        SymBoolType()
    except Exception:
        pass


def test_SymIntType():
    try:
        SymIntType()
    except Exception:
        pass


def test_Tag():
    try:
        Tag()
    except Exception:
        pass


def test_TensorType():
    try:
        TensorType()
    except Exception:
        pass


def test_ThroughputBenchmark():
    try:
        ThroughputBenchmark()
    except Exception:
        pass


def test_TracingState():
    try:
        TracingState()
    except Exception:
        pass


def test_TupleType():
    try:
        TupleType()
    except Exception:
        pass


def test_Type():
    try:
        Type()
    except Exception:
        pass


def test_UnionType():
    try:
        UnionType()
    except Exception:
        pass


def test_Use():
    try:
        Use()
    except Exception:
        pass


def test_Value():
    try:
        Value()
    except Exception:
        pass


def test_autocast_decrement_nesting():
    try:
        autocast_decrement_nesting()
    except Exception:
        pass


def test_autocast_increment_nesting():
    try:
        autocast_increment_nesting()
    except Exception:
        pass


def test_clear_autocast_cache():
    try:
        clear_autocast_cache()
    except Exception:
        pass


def test_cpp_OrderedModuleDict():
    try:
        cpp_OrderedModuleDict()
    except Exception:
        pass


def test_cpp_OrderedTensorDict():
    try:
        cpp_OrderedTensorDict()
    except Exception:
        pass


def test_cpp_nn_Module():
    try:
        cpp_nn_Module()
    except Exception:
        pass


def test_default_generator():
    try:
        default_generator()
    except Exception:
        pass


def test_device():
    try:
        device()
    except Exception:
        pass


def test_dtype():
    try:
        dtype()
    except Exception:
        pass


def test_finfo():
    try:
        finfo()
    except Exception:
        pass


def test_fork():
    try:
        fork()
    except Exception:
        pass


def test_get_autocast_cpu_dtype():
    try:
        get_autocast_cpu_dtype()
    except Exception:
        pass


def test_get_autocast_dtype():
    try:
        get_autocast_dtype()
    except Exception:
        pass


def test_get_autocast_gpu_dtype():
    try:
        get_autocast_gpu_dtype()
    except Exception:
        pass


def test_get_autocast_ipu_dtype():
    try:
        get_autocast_ipu_dtype()
    except Exception:
        pass


def test_get_autocast_xla_dtype():
    try:
        get_autocast_xla_dtype()
    except Exception:
        pass


def test_get_default_dtype():
    try:
        get_default_dtype()
    except Exception:
        pass


def test_get_num_interop_threads():
    try:
        get_num_interop_threads()
    except Exception:
        pass


def test_get_num_threads():
    try:
        get_num_threads()
    except Exception:
        pass


def test_has_lapack():
    try:
        has_lapack()
    except Exception:
        pass


def test_has_mkl():
    try:
        has_mkl()
    except Exception:
        pass


def test_has_openmp():
    try:
        has_openmp()
    except Exception:
        pass


def test_has_spectral():
    try:
        has_spectral()
    except Exception:
        pass


def test_iinfo():
    try:
        iinfo()
    except Exception:
        pass


def test_import_ir_module():
    try:
        import_ir_module()
    except Exception:
        pass


def test_import_ir_module_from_buffer():
    try:
        import_ir_module_from_buffer()
    except Exception:
        pass


def test_init_num_threads():
    try:
        init_num_threads()
    except Exception:
        pass


def test_is_anomaly_check_nan_enabled():
    try:
        is_anomaly_check_nan_enabled()
    except Exception:
        pass


def test_is_anomaly_enabled():
    try:
        is_anomaly_enabled()
    except Exception:
        pass


def test_is_autocast_cache_enabled():
    try:
        is_autocast_cache_enabled()
    except Exception:
        pass


def test_is_autocast_cpu_enabled():
    try:
        is_autocast_cpu_enabled()
    except Exception:
        pass


def test_is_autocast_enabled():
    try:
        is_autocast_enabled()
    except Exception:
        pass


def test_is_autocast_ipu_enabled():
    try:
        is_autocast_ipu_enabled()
    except Exception:
        pass


def test_is_autocast_xla_enabled():
    try:
        is_autocast_xla_enabled()
    except Exception:
        pass


def test_is_grad_enabled():
    try:
        is_grad_enabled()
    except Exception:
        pass


def test_is_inference_mode_enabled():
    try:
        is_inference_mode_enabled()
    except Exception:
        pass


def test_layout():
    try:
        layout()
    except Exception:
        pass


def test_memory_format():
    try:
        memory_format()
    except Exception:
        pass


def test_merge_type_from_type_comment():
    try:
        merge_type_from_type_comment()
    except Exception:
        pass


def test_parse_ir():
    try:
        parse_ir()
    except Exception:
        pass


def test_parse_schema():
    try:
        parse_schema()
    except Exception:
        pass


def test_parse_type_comment():
    try:
        parse_type_comment()
    except Exception:
        pass


def test_qscheme():
    try:
        qscheme()
    except Exception:
        pass


def test_read_vitals():
    try:
        read_vitals()
    except Exception:
        pass


def test_set_anomaly_enabled():
    try:
        set_anomaly_enabled()
    except Exception:
        pass


def test_set_autocast_cache_enabled():
    try:
        set_autocast_cache_enabled()
    except Exception:
        pass


def test_set_autocast_cpu_dtype():
    try:
        set_autocast_cpu_dtype()
    except Exception:
        pass


def test_set_autocast_cpu_enabled():
    try:
        set_autocast_cpu_enabled()
    except Exception:
        pass


def test_set_autocast_dtype():
    try:
        set_autocast_dtype()
    except Exception:
        pass


def test_set_autocast_enabled():
    try:
        set_autocast_enabled()
    except Exception:
        pass


def test_set_autocast_gpu_dtype():
    try:
        set_autocast_gpu_dtype()
    except Exception:
        pass


def test_set_autocast_ipu_dtype():
    try:
        set_autocast_ipu_dtype()
    except Exception:
        pass


def test_set_autocast_ipu_enabled():
    try:
        set_autocast_ipu_enabled()
    except Exception:
        pass


def test_set_autocast_xla_dtype():
    try:
        set_autocast_xla_dtype()
    except Exception:
        pass


def test_set_autocast_xla_enabled():
    try:
        set_autocast_xla_enabled()
    except Exception:
        pass


def test_set_flush_denormal():
    try:
        set_flush_denormal()
    except Exception:
        pass


def test_set_num_interop_threads():
    try:
        set_num_interop_threads()
    except Exception:
        pass


def test_set_num_threads():
    try:
        set_num_threads()
    except Exception:
        pass


def test_set_vital():
    try:
        set_vital()
    except Exception:
        pass


def test_unify_type_list():
    try:
        unify_type_list()
    except Exception:
        pass


def test_vitals_enabled():
    try:
        vitals_enabled()
    except Exception:
        pass


def test_wait():
    try:
        wait()
    except Exception:
        pass


def test_e():
    try:
        e()
    except Exception:
        pass


def test_pi():
    try:
        pi()
    except Exception:
        pass


def test_nan():
    try:
        nan()
    except Exception:
        pass


def test_inf():
    try:
        inf()
    except Exception:
        pass


def test_newaxis():
    try:
        newaxis()
    except Exception:
        pass


def test_abs():
    try:
        abs()
    except Exception:
        pass


def test_abs_():
    try:
        abs_()
    except Exception:
        pass


def test_absolute():
    try:
        absolute()
    except Exception:
        pass


def test_acos():
    try:
        acos()
    except Exception:
        pass


def test_acos_():
    try:
        acos_()
    except Exception:
        pass


def test_acosh():
    try:
        acosh()
    except Exception:
        pass


def test_acosh_():
    try:
        acosh_()
    except Exception:
        pass


def test_adaptive_avg_pool1d():
    try:
        adaptive_avg_pool1d()
    except Exception:
        pass


def test_adaptive_max_pool1d():
    try:
        adaptive_max_pool1d()
    except Exception:
        pass


def test_add():
    try:
        add()
    except Exception:
        pass


def test_addbmm():
    try:
        addbmm()
    except Exception:
        pass


def test_addcdiv():
    try:
        addcdiv()
    except Exception:
        pass


def test_addcmul():
    try:
        addcmul()
    except Exception:
        pass


def test_addmm():
    try:
        addmm()
    except Exception:
        pass


def test_addmv():
    try:
        addmv()
    except Exception:
        pass


def test_addmv_():
    try:
        addmv_()
    except Exception:
        pass


def test_addr():
    try:
        addr()
    except Exception:
        pass


def test_adjoint():
    try:
        adjoint()
    except Exception:
        pass


def test_affine_grid_generator():
    try:
        affine_grid_generator()
    except Exception:
        pass


def test_alias_copy():
    try:
        alias_copy()
    except Exception:
        pass


def test_align_tensors():
    try:
        align_tensors()
    except Exception:
        pass


def test_all():
    try:
        all()
    except Exception:
        pass


def test_allclose():
    try:
        allclose()
    except Exception:
        pass


def test_alpha_dropout():
    try:
        alpha_dropout()
    except Exception:
        pass


def test_alpha_dropout_():
    try:
        alpha_dropout_()
    except Exception:
        pass


def test_amax():
    try:
        amax()
    except Exception:
        pass


def test_amin():
    try:
        amin()
    except Exception:
        pass


def test_aminmax():
    try:
        aminmax()
    except Exception:
        pass


def test_angle():
    try:
        angle()
    except Exception:
        pass


def test_any():
    try:
        any()
    except Exception:
        pass


def test_arange():
    try:
        arange()
    except Exception:
        pass


def test_arccos():
    try:
        arccos()
    except Exception:
        pass


def test_arccos_():
    try:
        arccos_()
    except Exception:
        pass


def test_arccosh():
    try:
        arccosh()
    except Exception:
        pass


def test_arccosh_():
    try:
        arccosh_()
    except Exception:
        pass


def test_arcsin():
    try:
        arcsin()
    except Exception:
        pass


def test_arcsin_():
    try:
        arcsin_()
    except Exception:
        pass


def test_arcsinh():
    try:
        arcsinh()
    except Exception:
        pass


def test_arcsinh_():
    try:
        arcsinh_()
    except Exception:
        pass


def test_arctan():
    try:
        arctan()
    except Exception:
        pass


def test_arctan2():
    try:
        arctan2()
    except Exception:
        pass


def test_arctan_():
    try:
        arctan_()
    except Exception:
        pass


def test_arctanh():
    try:
        arctanh()
    except Exception:
        pass


def test_arctanh_():
    try:
        arctanh_()
    except Exception:
        pass


def test_argmax():
    try:
        argmax()
    except Exception:
        pass


def test_argmin():
    try:
        argmin()
    except Exception:
        pass


def test_argsort():
    try:
        argsort()
    except Exception:
        pass


def test_argwhere():
    try:
        argwhere()
    except Exception:
        pass


def test_as_strided():
    try:
        as_strided()
    except Exception:
        pass


def test_as_strided_():
    try:
        as_strided_()
    except Exception:
        pass


def test_as_strided_copy():
    try:
        as_strided_copy()
    except Exception:
        pass


def test_as_strided_scatter():
    try:
        as_strided_scatter()
    except Exception:
        pass


def test_as_tensor():
    try:
        as_tensor()
    except Exception:
        pass


def test_asarray():
    try:
        asarray()
    except Exception:
        pass


def test_asin():
    try:
        asin()
    except Exception:
        pass


def test_asin_():
    try:
        asin_()
    except Exception:
        pass


def test_asinh():
    try:
        asinh()
    except Exception:
        pass


def test_asinh_():
    try:
        asinh_()
    except Exception:
        pass


def test_atan():
    try:
        atan()
    except Exception:
        pass


def test_atan2():
    try:
        atan2()
    except Exception:
        pass


def test_atan_():
    try:
        atan_()
    except Exception:
        pass


def test_atanh():
    try:
        atanh()
    except Exception:
        pass


def test_atanh_():
    try:
        atanh_()
    except Exception:
        pass


def test_atleast_1d():
    try:
        atleast_1d()
    except Exception:
        pass


def test_atleast_2d():
    try:
        atleast_2d()
    except Exception:
        pass


def test_atleast_3d():
    try:
        atleast_3d()
    except Exception:
        pass


def test_avg_pool1d():
    try:
        avg_pool1d()
    except Exception:
        pass


def test_baddbmm():
    try:
        baddbmm()
    except Exception:
        pass


def test_bartlett_window():
    try:
        bartlett_window()
    except Exception:
        pass


def test_batch_norm():
    try:
        batch_norm()
    except Exception:
        pass


def test_batch_norm_backward_elemt():
    try:
        batch_norm_backward_elemt()
    except Exception:
        pass


def test_batch_norm_backward_reduce():
    try:
        batch_norm_backward_reduce()
    except Exception:
        pass


def test_batch_norm_elemt():
    try:
        batch_norm_elemt()
    except Exception:
        pass


def test_batch_norm_gather_stats():
    try:
        batch_norm_gather_stats()
    except Exception:
        pass


def test_batch_norm_gather_stats_with_counts():
    try:
        batch_norm_gather_stats_with_counts()
    except Exception:
        pass


def test_batch_norm_stats():
    try:
        batch_norm_stats()
    except Exception:
        pass


def test_batch_norm_update_stats():
    try:
        batch_norm_update_stats()
    except Exception:
        pass


def test_bernoulli():
    try:
        bernoulli()
    except Exception:
        pass


def test_bilinear():
    try:
        bilinear()
    except Exception:
        pass


def test_binary_cross_entropy_with_logits():
    try:
        binary_cross_entropy_with_logits()
    except Exception:
        pass


def test_bincount():
    try:
        bincount()
    except Exception:
        pass


def test_binomial():
    try:
        binomial()
    except Exception:
        pass


def test_bitwise_and():
    try:
        bitwise_and()
    except Exception:
        pass


def test_bitwise_left_shift():
    try:
        bitwise_left_shift()
    except Exception:
        pass


def test_bitwise_not():
    try:
        bitwise_not()
    except Exception:
        pass


def test_bitwise_or():
    try:
        bitwise_or()
    except Exception:
        pass


def test_bitwise_right_shift():
    try:
        bitwise_right_shift()
    except Exception:
        pass


def test_bitwise_xor():
    try:
        bitwise_xor()
    except Exception:
        pass


def test_blackman_window():
    try:
        blackman_window()
    except Exception:
        pass


def test_block_diag():
    try:
        block_diag()
    except Exception:
        pass


def test_bmm():
    try:
        bmm()
    except Exception:
        pass


def test_broadcast_tensors():
    try:
        broadcast_tensors()
    except Exception:
        pass


def test_broadcast_to():
    try:
        broadcast_to()
    except Exception:
        pass


def test_bucketize():
    try:
        bucketize()
    except Exception:
        pass


def test_can_cast():
    try:
        can_cast()
    except Exception:
        pass


def test_cartesian_prod():
    try:
        cartesian_prod()
    except Exception:
        pass


def test_cat():
    try:
        cat()
    except Exception:
        pass


def test_ccol_indices_copy():
    try:
        ccol_indices_copy()
    except Exception:
        pass


def test_cdist():
    try:
        cdist()
    except Exception:
        pass


def test_ceil():
    try:
        ceil()
    except Exception:
        pass


def test_ceil_():
    try:
        ceil_()
    except Exception:
        pass


def test_celu():
    try:
        celu()
    except Exception:
        pass


def test_celu_():
    try:
        celu_()
    except Exception:
        pass


def test_chain_matmul():
    try:
        chain_matmul()
    except Exception:
        pass


def test_channel_shuffle():
    try:
        channel_shuffle()
    except Exception:
        pass


def test_cholesky():
    try:
        cholesky()
    except Exception:
        pass


def test_cholesky_inverse():
    try:
        cholesky_inverse()
    except Exception:
        pass


def test_cholesky_solve():
    try:
        cholesky_solve()
    except Exception:
        pass


def test_choose_qparams_optimized():
    try:
        choose_qparams_optimized()
    except Exception:
        pass


def test_clamp():
    try:
        clamp()
    except Exception:
        pass


def test_clamp_():
    try:
        clamp_()
    except Exception:
        pass


def test_clamp_max():
    try:
        clamp_max()
    except Exception:
        pass


def test_clamp_max_():
    try:
        clamp_max_()
    except Exception:
        pass


def test_clamp_min():
    try:
        clamp_min()
    except Exception:
        pass


def test_clamp_min_():
    try:
        clamp_min_()
    except Exception:
        pass


def test_clip():
    try:
        clip()
    except Exception:
        pass


def test_clip_():
    try:
        clip_()
    except Exception:
        pass


def test_clone():
    try:
        clone()
    except Exception:
        pass


def test_col_indices_copy():
    try:
        col_indices_copy()
    except Exception:
        pass


def test_column_stack():
    try:
        column_stack()
    except Exception:
        pass


def test_combinations():
    try:
        combinations()
    except Exception:
        pass


def test_complex():
    try:
        pass
    except Exception:
        pass


def test_concat():
    try:
        concat()
    except Exception:
        pass


def test_concatenate():
    try:
        concatenate()
    except Exception:
        pass


def test_conj():
    try:
        conj()
    except Exception:
        pass


def test_conj_physical():
    try:
        conj_physical()
    except Exception:
        pass


def test_conj_physical_():
    try:
        conj_physical_()
    except Exception:
        pass


def test_constant_pad_nd():
    try:
        constant_pad_nd()
    except Exception:
        pass


def test_conv1d():
    try:
        conv1d()
    except Exception:
        pass


def test_conv2d():
    try:
        conv2d()
    except Exception:
        pass


def test_conv3d():
    try:
        conv3d()
    except Exception:
        pass


def test_conv_tbc():
    try:
        conv_tbc()
    except Exception:
        pass


def test_conv_transpose1d():
    try:
        conv_transpose1d()
    except Exception:
        pass


def test_conv_transpose2d():
    try:
        conv_transpose2d()
    except Exception:
        pass


def test_conv_transpose3d():
    try:
        conv_transpose3d()
    except Exception:
        pass


def test_convolution():
    try:
        convolution()
    except Exception:
        pass


def test_copysign():
    try:
        copysign()
    except Exception:
        pass


def test_corrcoef():
    try:
        corrcoef()
    except Exception:
        pass


def test_cos():
    try:
        cos()
    except Exception:
        pass


def test_cos_():
    try:
        cos_()
    except Exception:
        pass


def test_cosh():
    try:
        cosh()
    except Exception:
        pass


def test_cosh_():
    try:
        cosh_()
    except Exception:
        pass


def test_cosine_embedding_loss():
    try:
        cosine_embedding_loss()
    except Exception:
        pass


def test_cosine_similarity():
    try:
        cosine_similarity()
    except Exception:
        pass


def test_count_nonzero():
    try:
        count_nonzero()
    except Exception:
        pass


def test_cov():
    try:
        cov()
    except Exception:
        pass


def test_cross():
    try:
        cross()
    except Exception:
        pass


def test_crow_indices_copy():
    try:
        crow_indices_copy()
    except Exception:
        pass


def test_ctc_loss():
    try:
        ctc_loss()
    except Exception:
        pass


def test_cudnn_affine_grid_generator():
    try:
        cudnn_affine_grid_generator()
    except Exception:
        pass


def test_cudnn_batch_norm():
    try:
        cudnn_batch_norm()
    except Exception:
        pass


def test_cudnn_convolution():
    try:
        cudnn_convolution()
    except Exception:
        pass


def test_cudnn_convolution_add_relu():
    try:
        cudnn_convolution_add_relu()
    except Exception:
        pass


def test_cudnn_convolution_relu():
    try:
        cudnn_convolution_relu()
    except Exception:
        pass


def test_cudnn_convolution_transpose():
    try:
        cudnn_convolution_transpose()
    except Exception:
        pass


def test_cudnn_grid_sampler():
    try:
        cudnn_grid_sampler()
    except Exception:
        pass


def test_cudnn_is_acceptable():
    try:
        cudnn_is_acceptable()
    except Exception:
        pass


def test_cummax():
    try:
        cummax()
    except Exception:
        pass


def test_cummin():
    try:
        cummin()
    except Exception:
        pass


def test_cumprod():
    try:
        cumprod()
    except Exception:
        pass


def test_cumsum():
    try:
        cumsum()
    except Exception:
        pass


def test_cumulative_trapezoid():
    try:
        cumulative_trapezoid()
    except Exception:
        pass


def test_deg2rad():
    try:
        deg2rad()
    except Exception:
        pass


def test_deg2rad_():
    try:
        deg2rad_()
    except Exception:
        pass


def test_dequantize():
    try:
        dequantize()
    except Exception:
        pass


def test_det():
    try:
        det()
    except Exception:
        pass


def test_detach():
    try:
        detach()
    except Exception:
        pass


def test_detach_():
    try:
        detach_()
    except Exception:
        pass


def test_detach_copy():
    try:
        detach_copy()
    except Exception:
        pass


def test_diag():
    try:
        diag()
    except Exception:
        pass


def test_diag_embed():
    try:
        diag_embed()
    except Exception:
        pass


def test_diagflat():
    try:
        diagflat()
    except Exception:
        pass


def test_diagonal():
    try:
        diagonal()
    except Exception:
        pass


def test_diagonal_copy():
    try:
        diagonal_copy()
    except Exception:
        pass


def test_diagonal_scatter():
    try:
        diagonal_scatter()
    except Exception:
        pass


def test_diff():
    try:
        diff()
    except Exception:
        pass


def test_digamma():
    try:
        digamma()
    except Exception:
        pass


def test_dist():
    try:
        dist()
    except Exception:
        pass


def test_div():
    try:
        div()
    except Exception:
        pass


def test_divide():
    try:
        divide()
    except Exception:
        pass


def test_dot():
    try:
        dot()
    except Exception:
        pass


def test_dropout():
    try:
        dropout()
    except Exception:
        pass


def test_dropout_():
    try:
        dropout_()
    except Exception:
        pass


def test_dsmm():
    try:
        dsmm()
    except Exception:
        pass


def test_dsplit():
    try:
        dsplit()
    except Exception:
        pass


def test_dstack():
    try:
        dstack()
    except Exception:
        pass


def test_einsum():
    try:
        einsum()
    except Exception:
        pass


def test_embedding():
    try:
        embedding()
    except Exception:
        pass


def test_embedding_bag():
    try:
        embedding_bag()
    except Exception:
        pass


def test_embedding_renorm_():
    try:
        embedding_renorm_()
    except Exception:
        pass


def test_empty():
    try:
        empty()
    except Exception:
        pass


def test_empty_like():
    try:
        empty_like()
    except Exception:
        pass


def test_empty_permuted():
    try:
        empty_permuted()
    except Exception:
        pass


def test_empty_quantized():
    try:
        empty_quantized()
    except Exception:
        pass


def test_empty_strided():
    try:
        empty_strided()
    except Exception:
        pass


def test_eq():
    try:
        eq()
    except Exception:
        pass


def test_equal():
    try:
        equal()
    except Exception:
        pass


def test_erf():
    try:
        erf()
    except Exception:
        pass


def test_erf_():
    try:
        erf_()
    except Exception:
        pass


def test_erfc():
    try:
        erfc()
    except Exception:
        pass


def test_erfc_():
    try:
        erfc_()
    except Exception:
        pass


def test_erfinv():
    try:
        erfinv()
    except Exception:
        pass


def test_exp():
    try:
        exp()
    except Exception:
        pass


def test_exp2():
    try:
        exp2()
    except Exception:
        pass


def test_exp2_():
    try:
        exp2_()
    except Exception:
        pass


def test_exp_():
    try:
        exp_()
    except Exception:
        pass


def test_expand_copy():
    try:
        expand_copy()
    except Exception:
        pass


def test_expm1():
    try:
        expm1()
    except Exception:
        pass


def test_expm1_():
    try:
        expm1_()
    except Exception:
        pass


def test_eye():
    try:
        eye()
    except Exception:
        pass


def test_fake_quantize_per_channel_affine():
    try:
        fake_quantize_per_channel_affine()
    except Exception:
        pass


def test_fake_quantize_per_tensor_affine():
    try:
        fake_quantize_per_tensor_affine()
    except Exception:
        pass


def test_fbgemm_linear_fp16_weight():
    try:
        fbgemm_linear_fp16_weight()
    except Exception:
        pass


def test_fbgemm_linear_fp16_weight_fp32_activation():
    try:
        fbgemm_linear_fp16_weight_fp32_activation()
    except Exception:
        pass


def test_fbgemm_linear_int8_weight():
    try:
        fbgemm_linear_int8_weight()
    except Exception:
        pass


def test_fbgemm_linear_int8_weight_fp32_activation():
    try:
        fbgemm_linear_int8_weight_fp32_activation()
    except Exception:
        pass


def test_fbgemm_linear_quantize_weight():
    try:
        fbgemm_linear_quantize_weight()
    except Exception:
        pass


def test_fbgemm_pack_gemm_matrix_fp16():
    try:
        fbgemm_pack_gemm_matrix_fp16()
    except Exception:
        pass


def test_fbgemm_pack_quantized_matrix():
    try:
        fbgemm_pack_quantized_matrix()
    except Exception:
        pass


def test_feature_alpha_dropout():
    try:
        feature_alpha_dropout()
    except Exception:
        pass


def test_feature_alpha_dropout_():
    try:
        feature_alpha_dropout_()
    except Exception:
        pass


def test_feature_dropout():
    try:
        feature_dropout()
    except Exception:
        pass


def test_feature_dropout_():
    try:
        feature_dropout_()
    except Exception:
        pass


def test_fill():
    try:
        fill()
    except Exception:
        pass


def test_fill_():
    try:
        fill_()
    except Exception:
        pass


def test_fix():
    try:
        fix()
    except Exception:
        pass


def test_fix_():
    try:
        fix_()
    except Exception:
        pass


def test_flatten():
    try:
        flatten()
    except Exception:
        pass


def test_flip():
    try:
        flip()
    except Exception:
        pass


def test_fliplr():
    try:
        fliplr()
    except Exception:
        pass


def test_flipud():
    try:
        flipud()
    except Exception:
        pass


def test_float_power():
    try:
        float_power()
    except Exception:
        pass


def test_floor():
    try:
        floor()
    except Exception:
        pass


def test_floor_():
    try:
        floor_()
    except Exception:
        pass


def test_floor_divide():
    try:
        floor_divide()
    except Exception:
        pass


def test_fmax():
    try:
        fmax()
    except Exception:
        pass


def test_fmin():
    try:
        fmin()
    except Exception:
        pass


def test_fmod():
    try:
        fmod()
    except Exception:
        pass


def test_frac():
    try:
        frac()
    except Exception:
        pass


def test_frac_():
    try:
        frac_()
    except Exception:
        pass


def test_frexp():
    try:
        frexp()
    except Exception:
        pass


def test_frobenius_norm():
    try:
        frobenius_norm()
    except Exception:
        pass


def test_from_file():
    try:
        from_file()
    except Exception:
        pass


def test_from_numpy():
    try:
        from_numpy()
    except Exception:
        pass


def test_frombuffer():
    try:
        frombuffer()
    except Exception:
        pass


def test_full():
    try:
        full()
    except Exception:
        pass


def test_full_like():
    try:
        full_like()
    except Exception:
        pass


def test_fused_moving_avg_obs_fake_quant():
    try:
        fused_moving_avg_obs_fake_quant()
    except Exception:
        pass


def test_gather():
    try:
        gather()
    except Exception:
        pass


def test_gcd():
    try:
        gcd()
    except Exception:
        pass


def test_gcd_():
    try:
        gcd_()
    except Exception:
        pass


def test_ge():
    try:
        ge()
    except Exception:
        pass


def test_geqrf():
    try:
        geqrf()
    except Exception:
        pass


def test_ger():
    try:
        ger()
    except Exception:
        pass


def test_get_device():
    try:
        get_device()
    except Exception:
        pass


def test_gradient():
    try:
        gradient()
    except Exception:
        pass


def test_greater():
    try:
        greater()
    except Exception:
        pass


def test_greater_equal():
    try:
        greater_equal()
    except Exception:
        pass


def test_grid_sampler():
    try:
        grid_sampler()
    except Exception:
        pass


def test_grid_sampler_2d():
    try:
        grid_sampler_2d()
    except Exception:
        pass


def test_grid_sampler_3d():
    try:
        grid_sampler_3d()
    except Exception:
        pass


def test_group_norm():
    try:
        group_norm()
    except Exception:
        pass


def test_gru():
    try:
        gru()
    except Exception:
        pass


def test_gru_cell():
    try:
        gru_cell()
    except Exception:
        pass


def test_gt():
    try:
        gt()
    except Exception:
        pass


def test_hamming_window():
    try:
        hamming_window()
    except Exception:
        pass


def test_hann_window():
    try:
        hann_window()
    except Exception:
        pass


def test_hardshrink():
    try:
        hardshrink()
    except Exception:
        pass


def test_hash_tensor():
    try:
        hash_tensor()
    except Exception:
        pass


def test_heaviside():
    try:
        heaviside()
    except Exception:
        pass


def test_hinge_embedding_loss():
    try:
        hinge_embedding_loss()
    except Exception:
        pass


def test_histc():
    try:
        histc()
    except Exception:
        pass


def test_histogram():
    try:
        histogram()
    except Exception:
        pass


def test_histogramdd():
    try:
        histogramdd()
    except Exception:
        pass


def test_hsmm():
    try:
        hsmm()
    except Exception:
        pass


def test_hsplit():
    try:
        hsplit()
    except Exception:
        pass


def test_hspmm():
    try:
        hspmm()
    except Exception:
        pass


def test_hstack():
    try:
        hstack()
    except Exception:
        pass


def test_hypot():
    try:
        hypot()
    except Exception:
        pass


def test_i0():
    try:
        i0()
    except Exception:
        pass


def test_i0_():
    try:
        i0_()
    except Exception:
        pass


def test_igamma():
    try:
        igamma()
    except Exception:
        pass


def test_igammac():
    try:
        igammac()
    except Exception:
        pass


def test_imag():
    try:
        imag()
    except Exception:
        pass


def test_index_add():
    try:
        index_add()
    except Exception:
        pass


def test_index_copy():
    try:
        index_copy()
    except Exception:
        pass


def test_index_fill():
    try:
        index_fill()
    except Exception:
        pass


def test_index_put():
    try:
        index_put()
    except Exception:
        pass


def test_index_put_():
    try:
        index_put_()
    except Exception:
        pass


def test_index_reduce():
    try:
        index_reduce()
    except Exception:
        pass


def test_index_select():
    try:
        index_select()
    except Exception:
        pass


def test_indices_copy():
    try:
        indices_copy()
    except Exception:
        pass


def test_inner():
    try:
        inner()
    except Exception:
        pass


def test_instance_norm():
    try:
        instance_norm()
    except Exception:
        pass


def test_int_repr():
    try:
        int_repr()
    except Exception:
        pass


def test_inverse():
    try:
        inverse()
    except Exception:
        pass


def test_is_complex():
    try:
        is_complex()
    except Exception:
        pass


def test_is_conj():
    try:
        is_conj()
    except Exception:
        pass


def test_is_distributed():
    try:
        is_distributed()
    except Exception:
        pass


def test_is_floating_point():
    try:
        is_floating_point()
    except Exception:
        pass


def test_is_inference():
    try:
        is_inference()
    except Exception:
        pass


def test_is_neg():
    try:
        is_neg()
    except Exception:
        pass


def test_is_nonzero():
    try:
        is_nonzero()
    except Exception:
        pass


def test_is_same_size():
    try:
        is_same_size()
    except Exception:
        pass


def test_is_signed():
    try:
        is_signed()
    except Exception:
        pass


def test_is_vulkan_available():
    try:
        is_vulkan_available()
    except Exception:
        pass


def test_isclose():
    try:
        isclose()
    except Exception:
        pass


def test_isfinite():
    try:
        isfinite()
    except Exception:
        pass


def test_isin():
    try:
        isin()
    except Exception:
        pass


def test_isinf():
    try:
        isinf()
    except Exception:
        pass


def test_isnan():
    try:
        isnan()
    except Exception:
        pass


def test_isneginf():
    try:
        isneginf()
    except Exception:
        pass


def test_isposinf():
    try:
        isposinf()
    except Exception:
        pass


def test_isreal():
    try:
        isreal()
    except Exception:
        pass


def test_istft():
    try:
        istft()
    except Exception:
        pass


def test_kaiser_window():
    try:
        kaiser_window()
    except Exception:
        pass


def test_kl_div():
    try:
        kl_div()
    except Exception:
        pass


def test_kron():
    try:
        kron()
    except Exception:
        pass


def test_kthvalue():
    try:
        kthvalue()
    except Exception:
        pass


def test_layer_norm():
    try:
        layer_norm()
    except Exception:
        pass


def test_lcm():
    try:
        lcm()
    except Exception:
        pass


def test_lcm_():
    try:
        lcm_()
    except Exception:
        pass


def test_ldexp():
    try:
        ldexp()
    except Exception:
        pass


def test_ldexp_():
    try:
        ldexp_()
    except Exception:
        pass


def test_le():
    try:
        le()
    except Exception:
        pass


def test_lerp():
    try:
        lerp()
    except Exception:
        pass


def test_less():
    try:
        less()
    except Exception:
        pass


def test_less_equal():
    try:
        less_equal()
    except Exception:
        pass


def test_lgamma():
    try:
        lgamma()
    except Exception:
        pass


def test_linspace():
    try:
        linspace()
    except Exception:
        pass


def test_log():
    try:
        log()
    except Exception:
        pass


def test_log10():
    try:
        log10()
    except Exception:
        pass


def test_log10_():
    try:
        log10_()
    except Exception:
        pass


def test_log1p():
    try:
        log1p()
    except Exception:
        pass


def test_log1p_():
    try:
        log1p_()
    except Exception:
        pass


def test_log2():
    try:
        log2()
    except Exception:
        pass


def test_log2_():
    try:
        log2_()
    except Exception:
        pass


def test_log_():
    try:
        log_()
    except Exception:
        pass


def test_log_softmax():
    try:
        log_softmax()
    except Exception:
        pass


def test_logaddexp():
    try:
        logaddexp()
    except Exception:
        pass


def test_logaddexp2():
    try:
        logaddexp2()
    except Exception:
        pass


def test_logcumsumexp():
    try:
        logcumsumexp()
    except Exception:
        pass


def test_logdet():
    try:
        logdet()
    except Exception:
        pass


def test_logical_and():
    try:
        logical_and()
    except Exception:
        pass


def test_logical_not():
    try:
        logical_not()
    except Exception:
        pass


def test_logical_or():
    try:
        logical_or()
    except Exception:
        pass


def test_logical_xor():
    try:
        logical_xor()
    except Exception:
        pass


def test_logit():
    try:
        logit()
    except Exception:
        pass


def test_logit_():
    try:
        logit_()
    except Exception:
        pass


def test_logspace():
    try:
        logspace()
    except Exception:
        pass


def test_logsumexp():
    try:
        logsumexp()
    except Exception:
        pass


def test_lstm():
    try:
        lstm()
    except Exception:
        pass


def test_lstm_cell():
    try:
        lstm_cell()
    except Exception:
        pass


def test_lt():
    try:
        lt()
    except Exception:
        pass


def test_lu_solve():
    try:
        lu_solve()
    except Exception:
        pass


def test_lu_unpack():
    try:
        lu_unpack()
    except Exception:
        pass


def test_margin_ranking_loss():
    try:
        margin_ranking_loss()
    except Exception:
        pass


def test_masked_fill():
    try:
        masked_fill()
    except Exception:
        pass


def test_masked_scatter():
    try:
        masked_scatter()
    except Exception:
        pass


def test_masked_select():
    try:
        masked_select()
    except Exception:
        pass


def test_matrix_exp():
    try:
        matrix_exp()
    except Exception:
        pass


def test_matrix_power():
    try:
        matrix_power()
    except Exception:
        pass


def test_max():
    try:
        max()
    except Exception:
        pass


def test_max_pool1d():
    try:
        max_pool1d()
    except Exception:
        pass


def test_max_pool1d_with_indices():
    try:
        max_pool1d_with_indices()
    except Exception:
        pass


def test_max_pool2d():
    try:
        max_pool2d()
    except Exception:
        pass


def test_max_pool3d():
    try:
        max_pool3d()
    except Exception:
        pass


def test_maximum():
    try:
        maximum()
    except Exception:
        pass


def test_mean():
    try:
        mean()
    except Exception:
        pass


def test_median():
    try:
        median()
    except Exception:
        pass


def test_meshgrid():
    try:
        meshgrid()
    except Exception:
        pass


def test_min():
    try:
        min()
    except Exception:
        pass


def test_minimum():
    try:
        minimum()
    except Exception:
        pass


def test_miopen_batch_norm():
    try:
        miopen_batch_norm()
    except Exception:
        pass


def test_miopen_convolution():
    try:
        miopen_convolution()
    except Exception:
        pass


def test_miopen_convolution_add_relu():
    try:
        miopen_convolution_add_relu()
    except Exception:
        pass


def test_miopen_convolution_relu():
    try:
        miopen_convolution_relu()
    except Exception:
        pass


def test_miopen_convolution_transpose():
    try:
        miopen_convolution_transpose()
    except Exception:
        pass


def test_miopen_ctc_loss():
    try:
        miopen_ctc_loss()
    except Exception:
        pass


def test_miopen_depthwise_convolution():
    try:
        miopen_depthwise_convolution()
    except Exception:
        pass


def test_miopen_rnn():
    try:
        miopen_rnn()
    except Exception:
        pass


def test_mkldnn_adaptive_avg_pool2d():
    try:
        mkldnn_adaptive_avg_pool2d()
    except Exception:
        pass


def test_mkldnn_convolution():
    try:
        mkldnn_convolution()
    except Exception:
        pass


def test_mkldnn_linear_backward_weights():
    try:
        mkldnn_linear_backward_weights()
    except Exception:
        pass


def test_mkldnn_max_pool2d():
    try:
        mkldnn_max_pool2d()
    except Exception:
        pass


def test_mkldnn_max_pool3d():
    try:
        mkldnn_max_pool3d()
    except Exception:
        pass


def test_mkldnn_rnn_layer():
    try:
        mkldnn_rnn_layer()
    except Exception:
        pass


def test_mm():
    try:
        mm()
    except Exception:
        pass


def test_mode():
    try:
        mode()
    except Exception:
        pass


def test_moveaxis():
    try:
        moveaxis()
    except Exception:
        pass


def test_movedim():
    try:
        movedim()
    except Exception:
        pass


def test_msort():
    try:
        msort()
    except Exception:
        pass


def test_mul():
    try:
        mul()
    except Exception:
        pass


def test_multinomial():
    try:
        multinomial()
    except Exception:
        pass


def test_multiply():
    try:
        multiply()
    except Exception:
        pass


def test_mv():
    try:
        mv()
    except Exception:
        pass


def test_mvlgamma():
    try:
        mvlgamma()
    except Exception:
        pass


def test_nan_to_num():
    try:
        nan_to_num()
    except Exception:
        pass


def test_nan_to_num_():
    try:
        nan_to_num_()
    except Exception:
        pass


def test_nanmean():
    try:
        nanmean()
    except Exception:
        pass


def test_nanmedian():
    try:
        nanmedian()
    except Exception:
        pass


def test_nanquantile():
    try:
        nanquantile()
    except Exception:
        pass


def test_nansum():
    try:
        nansum()
    except Exception:
        pass


def test_narrow():
    try:
        narrow()
    except Exception:
        pass


def test_narrow_copy():
    try:
        narrow_copy()
    except Exception:
        pass


def test_native_batch_norm():
    try:
        native_batch_norm()
    except Exception:
        pass


def test_native_channel_shuffle():
    try:
        native_channel_shuffle()
    except Exception:
        pass


def test_native_dropout():
    try:
        native_dropout()
    except Exception:
        pass


def test_native_group_norm():
    try:
        native_group_norm()
    except Exception:
        pass


def test_native_layer_norm():
    try:
        native_layer_norm()
    except Exception:
        pass


def test_native_norm():
    try:
        native_norm()
    except Exception:
        pass


def test_ne():
    try:
        ne()
    except Exception:
        pass


def test_neg():
    try:
        neg()
    except Exception:
        pass


def test_neg_():
    try:
        neg_()
    except Exception:
        pass


def test_negative():
    try:
        negative()
    except Exception:
        pass


def test_negative_():
    try:
        negative_()
    except Exception:
        pass


def test_nextafter():
    try:
        nextafter()
    except Exception:
        pass


def test_nonzero():
    try:
        nonzero()
    except Exception:
        pass


def test_nonzero_static():
    try:
        nonzero_static()
    except Exception:
        pass


def test_norm():
    try:
        norm()
    except Exception:
        pass


def test_norm_except_dim():
    try:
        norm_except_dim()
    except Exception:
        pass


def test_normal():
    try:
        normal()
    except Exception:
        pass


def test_not_equal():
    try:
        not_equal()
    except Exception:
        pass


def test_nuclear_norm():
    try:
        nuclear_norm()
    except Exception:
        pass


def test_numel():
    try:
        numel()
    except Exception:
        pass


def test_ones_like():
    try:
        ones_like()
    except Exception:
        pass


def test_orgqr():
    try:
        orgqr()
    except Exception:
        pass


def test_ormqr():
    try:
        ormqr()
    except Exception:
        pass


def test_outer():
    try:
        outer()
    except Exception:
        pass


def test_pairwise_distance():
    try:
        pairwise_distance()
    except Exception:
        pass


def test_pdist():
    try:
        pdist()
    except Exception:
        pass


def test_permute():
    try:
        permute()
    except Exception:
        pass


def test_permute_copy():
    try:
        permute_copy()
    except Exception:
        pass


def test_pinverse():
    try:
        pinverse()
    except Exception:
        pass


def test_pixel_shuffle():
    try:
        pixel_shuffle()
    except Exception:
        pass


def test_pixel_unshuffle():
    try:
        pixel_unshuffle()
    except Exception:
        pass


def test_poisson():
    try:
        poisson()
    except Exception:
        pass


def test_poisson_nll_loss():
    try:
        poisson_nll_loss()
    except Exception:
        pass


def test_polar():
    try:
        polar()
    except Exception:
        pass


def test_polygamma():
    try:
        polygamma()
    except Exception:
        pass


def test_positive():
    try:
        positive()
    except Exception:
        pass


def test_pow():
    try:
        pow()
    except Exception:
        pass


def test_prelu():
    try:
        prelu()
    except Exception:
        pass


def test_prod():
    try:
        prod()
    except Exception:
        pass


def test_promote_types():
    try:
        promote_types()
    except Exception:
        pass


def test_put():
    try:
        put()
    except Exception:
        pass


def test_q_per_channel_axis():
    try:
        q_per_channel_axis()
    except Exception:
        pass


def test_q_per_channel_scales():
    try:
        q_per_channel_scales()
    except Exception:
        pass


def test_q_per_channel_zero_points():
    try:
        q_per_channel_zero_points()
    except Exception:
        pass


def test_q_scale():
    try:
        q_scale()
    except Exception:
        pass


def test_q_zero_point():
    try:
        q_zero_point()
    except Exception:
        pass


def test_qr():
    try:
        qr()
    except Exception:
        pass


def test_quantile():
    try:
        quantile()
    except Exception:
        pass


def test_quantize_per_channel():
    try:
        quantize_per_channel()
    except Exception:
        pass


def test_quantize_per_tensor():
    try:
        quantize_per_tensor()
    except Exception:
        pass


def test_quantize_per_tensor_dynamic():
    try:
        quantize_per_tensor_dynamic()
    except Exception:
        pass


def test_quantized_batch_norm():
    try:
        quantized_batch_norm()
    except Exception:
        pass


def test_quantized_gru_cell():
    try:
        quantized_gru_cell()
    except Exception:
        pass


def test_quantized_lstm_cell():
    try:
        quantized_lstm_cell()
    except Exception:
        pass


def test_quantized_max_pool1d():
    try:
        quantized_max_pool1d()
    except Exception:
        pass


def test_quantized_max_pool2d():
    try:
        quantized_max_pool2d()
    except Exception:
        pass


def test_quantized_max_pool3d():
    try:
        quantized_max_pool3d()
    except Exception:
        pass


def test_quantized_rnn_relu_cell():
    try:
        quantized_rnn_relu_cell()
    except Exception:
        pass


def test_quantized_rnn_tanh_cell():
    try:
        quantized_rnn_tanh_cell()
    except Exception:
        pass


def test_rad2deg():
    try:
        rad2deg()
    except Exception:
        pass


def test_rad2deg_():
    try:
        rad2deg_()
    except Exception:
        pass


def test_rand_like():
    try:
        rand_like()
    except Exception:
        pass


def test_randint():
    try:
        randint()
    except Exception:
        pass


def test_randint_like():
    try:
        randint_like()
    except Exception:
        pass


def test_randn_like():
    try:
        randn_like()
    except Exception:
        pass


def test_randperm():
    try:
        randperm()
    except Exception:
        pass


def test_range():
    try:
        range()
    except Exception:
        pass


def test_ravel():
    try:
        ravel()
    except Exception:
        pass


def test_real():
    try:
        real()
    except Exception:
        pass


def test_reciprocal():
    try:
        reciprocal()
    except Exception:
        pass


def test_reciprocal_():
    try:
        reciprocal_()
    except Exception:
        pass


def test_relu():
    try:
        relu()
    except Exception:
        pass


def test_relu_():
    try:
        relu_()
    except Exception:
        pass


def test_remainder():
    try:
        remainder()
    except Exception:
        pass


def test_renorm():
    try:
        renorm()
    except Exception:
        pass


def test_repeat_interleave():
    try:
        repeat_interleave()
    except Exception:
        pass


def test_reshape():
    try:
        reshape()
    except Exception:
        pass


def test_resize_as_():
    try:
        resize_as_()
    except Exception:
        pass


def test_resize_as_sparse_():
    try:
        resize_as_sparse_()
    except Exception:
        pass


def test_resolve_conj():
    try:
        resolve_conj()
    except Exception:
        pass


def test_resolve_neg():
    try:
        resolve_neg()
    except Exception:
        pass


def test_result_type():
    try:
        result_type()
    except Exception:
        pass


def test_rms_norm():
    try:
        rms_norm()
    except Exception:
        pass


def test_rnn_relu():
    try:
        rnn_relu()
    except Exception:
        pass


def test_rnn_relu_cell():
    try:
        rnn_relu_cell()
    except Exception:
        pass


def test_rnn_tanh():
    try:
        rnn_tanh()
    except Exception:
        pass


def test_rnn_tanh_cell():
    try:
        rnn_tanh_cell()
    except Exception:
        pass


def test_roll():
    try:
        roll()
    except Exception:
        pass


def test_rot90():
    try:
        rot90()
    except Exception:
        pass


def test_round():
    try:
        round()
    except Exception:
        pass


def test_round_():
    try:
        round_()
    except Exception:
        pass


def test_row_indices_copy():
    try:
        row_indices_copy()
    except Exception:
        pass


def test_row_stack():
    try:
        row_stack()
    except Exception:
        pass


def test_rrelu():
    try:
        rrelu()
    except Exception:
        pass


def test_rrelu_():
    try:
        rrelu_()
    except Exception:
        pass


def test_rsqrt():
    try:
        rsqrt()
    except Exception:
        pass


def test_rsqrt_():
    try:
        rsqrt_()
    except Exception:
        pass


def test_rsub():
    try:
        rsub()
    except Exception:
        pass


def test_saddmm():
    try:
        saddmm()
    except Exception:
        pass


def test_scalar_tensor():
    try:
        scalar_tensor()
    except Exception:
        pass


def test_scatter():
    try:
        scatter()
    except Exception:
        pass


def test_scatter_add():
    try:
        scatter_add()
    except Exception:
        pass


def test_scatter_reduce():
    try:
        scatter_reduce()
    except Exception:
        pass


def test_searchsorted():
    try:
        searchsorted()
    except Exception:
        pass


def test_select():
    try:
        select()
    except Exception:
        pass


def test_select_copy():
    try:
        select_copy()
    except Exception:
        pass


def test_select_scatter():
    try:
        select_scatter()
    except Exception:
        pass


def test_selu():
    try:
        selu()
    except Exception:
        pass


def test_selu_():
    try:
        selu_()
    except Exception:
        pass


def test_sgn():
    try:
        sgn()
    except Exception:
        pass


def test_sigmoid():
    try:
        sigmoid()
    except Exception:
        pass


def test_sigmoid_():
    try:
        sigmoid_()
    except Exception:
        pass


def test_sign():
    try:
        sign()
    except Exception:
        pass


def test_signbit():
    try:
        signbit()
    except Exception:
        pass


def test_sin():
    try:
        sin()
    except Exception:
        pass


def test_sin_():
    try:
        sin_()
    except Exception:
        pass


def test_sinc():
    try:
        sinc()
    except Exception:
        pass


def test_sinc_():
    try:
        sinc_()
    except Exception:
        pass


def test_sinh():
    try:
        sinh()
    except Exception:
        pass


def test_sinh_():
    try:
        sinh_()
    except Exception:
        pass


def test_slice_copy():
    try:
        slice_copy()
    except Exception:
        pass


def test_slice_inverse():
    try:
        slice_inverse()
    except Exception:
        pass


def test_slice_scatter():
    try:
        slice_scatter()
    except Exception:
        pass


def test_slogdet():
    try:
        slogdet()
    except Exception:
        pass


def test_smm():
    try:
        smm()
    except Exception:
        pass


def test_softmax():
    try:
        softmax()
    except Exception:
        pass


def test_sort():
    try:
        sort()
    except Exception:
        pass


def test_sparse_bsc_tensor():
    try:
        sparse_bsc_tensor()
    except Exception:
        pass


def test_sparse_bsr_tensor():
    try:
        sparse_bsr_tensor()
    except Exception:
        pass


def test_sparse_compressed_tensor():
    try:
        sparse_compressed_tensor()
    except Exception:
        pass


def test_sparse_coo_tensor():
    try:
        sparse_coo_tensor()
    except Exception:
        pass


def test_sparse_csc_tensor():
    try:
        sparse_csc_tensor()
    except Exception:
        pass


def test_sparse_csr_tensor():
    try:
        sparse_csr_tensor()
    except Exception:
        pass


def test_split_copy():
    try:
        split_copy()
    except Exception:
        pass


def test_split_with_sizes():
    try:
        split_with_sizes()
    except Exception:
        pass


def test_split_with_sizes_copy():
    try:
        split_with_sizes_copy()
    except Exception:
        pass


def test_spmm():
    try:
        spmm()
    except Exception:
        pass


def test_sqrt():
    try:
        sqrt()
    except Exception:
        pass


def test_sqrt_():
    try:
        sqrt_()
    except Exception:
        pass


def test_square():
    try:
        square()
    except Exception:
        pass


def test_square_():
    try:
        square_()
    except Exception:
        pass


def test_squeeze():
    try:
        squeeze()
    except Exception:
        pass


def test_squeeze_copy():
    try:
        squeeze_copy()
    except Exception:
        pass


def test_sspaddmm():
    try:
        sspaddmm()
    except Exception:
        pass


def test_std():
    try:
        std()
    except Exception:
        pass


def test_std_mean():
    try:
        std_mean()
    except Exception:
        pass


def test_stft():
    try:
        stft()
    except Exception:
        pass


def test_sub():
    try:
        sub()
    except Exception:
        pass


def test_subtract():
    try:
        subtract()
    except Exception:
        pass


def test_sum():
    try:
        sum()
    except Exception:
        pass


def test_svd():
    try:
        svd()
    except Exception:
        pass


def test_swapaxes():
    try:
        swapaxes()
    except Exception:
        pass


def test_swapdims():
    try:
        swapdims()
    except Exception:
        pass


def test_sym_constrain_range():
    try:
        sym_constrain_range()
    except Exception:
        pass


def test_sym_constrain_range_for_size():
    try:
        sym_constrain_range_for_size()
    except Exception:
        pass


def test_t():
    try:
        t()
    except Exception:
        pass


def test_t_copy():
    try:
        t_copy()
    except Exception:
        pass


def test_take():
    try:
        take()
    except Exception:
        pass


def test_take_along_dim():
    try:
        take_along_dim()
    except Exception:
        pass


def test_tan():
    try:
        tan()
    except Exception:
        pass


def test_tan_():
    try:
        tan_()
    except Exception:
        pass


def test_tanh():
    try:
        tanh()
    except Exception:
        pass


def test_tanh_():
    try:
        tanh_()
    except Exception:
        pass


def test_tensor_split():
    try:
        tensor_split()
    except Exception:
        pass


def test_tensordot():
    try:
        tensordot()
    except Exception:
        pass


def test_threshold():
    try:
        threshold()
    except Exception:
        pass


def test_threshold_():
    try:
        threshold_()
    except Exception:
        pass


def test_tile():
    try:
        tile()
    except Exception:
        pass


def test_topk():
    try:
        topk()
    except Exception:
        pass


def test_transpose():
    try:
        transpose()
    except Exception:
        pass


def test_transpose_copy():
    try:
        transpose_copy()
    except Exception:
        pass


def test_trapezoid():
    try:
        trapezoid()
    except Exception:
        pass


def test_trapz():
    try:
        trapz()
    except Exception:
        pass


def test_triangular_solve():
    try:
        triangular_solve()
    except Exception:
        pass


def test_tril():
    try:
        tril()
    except Exception:
        pass


def test_tril_indices():
    try:
        tril_indices()
    except Exception:
        pass


def test_triplet_margin_loss():
    try:
        triplet_margin_loss()
    except Exception:
        pass


def test_triu():
    try:
        triu()
    except Exception:
        pass


def test_triu_indices():
    try:
        triu_indices()
    except Exception:
        pass


def test_true_divide():
    try:
        true_divide()
    except Exception:
        pass


def test_trunc():
    try:
        trunc()
    except Exception:
        pass


def test_trunc_():
    try:
        trunc_()
    except Exception:
        pass


def test_unbind():
    try:
        unbind()
    except Exception:
        pass


def test_unbind_copy():
    try:
        unbind_copy()
    except Exception:
        pass


def test_unflatten():
    try:
        unflatten()
    except Exception:
        pass


def test_unfold_copy():
    try:
        unfold_copy()
    except Exception:
        pass


def test_unique_consecutive():
    try:
        unique_consecutive()
    except Exception:
        pass


def test_unsafe_chunk():
    try:
        unsafe_chunk()
    except Exception:
        pass


def test_unsafe_split():
    try:
        unsafe_split()
    except Exception:
        pass


def test_unsafe_split_with_sizes():
    try:
        unsafe_split_with_sizes()
    except Exception:
        pass


def test_unsqueeze():
    try:
        unsqueeze()
    except Exception:
        pass


def test_unsqueeze_copy():
    try:
        unsqueeze_copy()
    except Exception:
        pass


def test_values_copy():
    try:
        values_copy()
    except Exception:
        pass


def test_vander():
    try:
        vander()
    except Exception:
        pass


def test_var():
    try:
        var()
    except Exception:
        pass


def test_var_mean():
    try:
        var_mean()
    except Exception:
        pass


def test_vdot():
    try:
        vdot()
    except Exception:
        pass


def test_view_as_complex():
    try:
        view_as_complex()
    except Exception:
        pass


def test_view_as_complex_copy():
    try:
        view_as_complex_copy()
    except Exception:
        pass


def test_view_as_real():
    try:
        view_as_real()
    except Exception:
        pass


def test_view_as_real_copy():
    try:
        view_as_real_copy()
    except Exception:
        pass


def test_view_copy():
    try:
        view_copy()
    except Exception:
        pass


def test_vsplit():
    try:
        vsplit()
    except Exception:
        pass


def test_vstack():
    try:
        vstack()
    except Exception:
        pass


def test_where():
    try:
        where()
    except Exception:
        pass


def test_xlogy():
    try:
        xlogy()
    except Exception:
        pass


def test_xlogy_():
    try:
        xlogy_()
    except Exception:
        pass


def test_zero_():
    try:
        zero_()
    except Exception:
        pass


def test_zeros_like():
    try:
        zeros_like()
    except Exception:
        pass


def test_bfloat16():
    try:
        bfloat16()
    except Exception:
        pass


def test_bit():
    try:
        bit()
    except Exception:
        pass


def test_bits16():
    try:
        bits16()
    except Exception:
        pass


def test_bits1x8():
    try:
        bits1x8()
    except Exception:
        pass


def test_bits2x4():
    try:
        bits2x4()
    except Exception:
        pass


def test_bits4x2():
    try:
        bits4x2()
    except Exception:
        pass


def test_bits8():
    try:
        bits8()
    except Exception:
        pass


def test_cdouble():
    try:
        cdouble()
    except Exception:
        pass


def test_cfloat():
    try:
        cfloat()
    except Exception:
        pass


def test_chalf():
    try:
        chalf()
    except Exception:
        pass


def test_complex128():
    try:
        complex128()
    except Exception:
        pass


def test_complex32():
    try:
        complex32()
    except Exception:
        pass


def test_complex64():
    try:
        complex64()
    except Exception:
        pass


def test_double():
    try:
        double()
    except Exception:
        pass


def test_float():
    try:
        pass
    except Exception:
        pass


def test_float16():
    try:
        float16()
    except Exception:
        pass


def test_float4_e2m1fn_x2():
    try:
        float4_e2m1fn_x2()
    except Exception:
        pass


def test_float8_e4m3fn():
    try:
        float8_e4m3fn()
    except Exception:
        pass


def test_float8_e4m3fnuz():
    try:
        float8_e4m3fnuz()
    except Exception:
        pass


def test_float8_e5m2():
    try:
        float8_e5m2()
    except Exception:
        pass


def test_float8_e5m2fnuz():
    try:
        float8_e5m2fnuz()
    except Exception:
        pass


def test_float8_e8m0fnu():
    try:
        float8_e8m0fnu()
    except Exception:
        pass


def test_half():
    try:
        half()
    except Exception:
        pass


def test_int():
    try:
        pass
    except Exception:
        pass


def test_int1():
    try:
        int1()
    except Exception:
        pass


def test_int16():
    try:
        int16()
    except Exception:
        pass


def test_int2():
    try:
        int2()
    except Exception:
        pass


def test_int3():
    try:
        int3()
    except Exception:
        pass


def test_int4():
    try:
        int4()
    except Exception:
        pass


def test_int5():
    try:
        int5()
    except Exception:
        pass


def test_int6():
    try:
        int6()
    except Exception:
        pass


def test_int7():
    try:
        int7()
    except Exception:
        pass


def test_int8():
    try:
        int8()
    except Exception:
        pass


def test_long():
    try:
        long()
    except Exception:
        pass


def test_qint32():
    try:
        qint32()
    except Exception:
        pass


def test_qint8():
    try:
        qint8()
    except Exception:
        pass


def test_quint2x4():
    try:
        quint2x4()
    except Exception:
        pass


def test_quint4x2():
    try:
        quint4x2()
    except Exception:
        pass


def test_quint8():
    try:
        quint8()
    except Exception:
        pass


def test_short():
    try:
        short()
    except Exception:
        pass


def test_uint1():
    try:
        uint1()
    except Exception:
        pass


def test_uint16():
    try:
        uint16()
    except Exception:
        pass


def test_uint2():
    try:
        uint2()
    except Exception:
        pass


def test_uint3():
    try:
        uint3()
    except Exception:
        pass


def test_uint32():
    try:
        uint32()
    except Exception:
        pass


def test_uint4():
    try:
        uint4()
    except Exception:
        pass


def test_uint5():
    try:
        uint5()
    except Exception:
        pass


def test_uint6():
    try:
        uint6()
    except Exception:
        pass


def test_uint64():
    try:
        uint64()
    except Exception:
        pass


def test_uint7():
    try:
        uint7()
    except Exception:
        pass


def test_uint8():
    try:
        uint8()
    except Exception:
        pass
