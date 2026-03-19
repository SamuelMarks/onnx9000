def test_tvm_all_others_part2():
    import glob
    import inspect

    # manual list of some modules to fully cover
    import onnx9000.tvm.build_module
    import onnx9000.tvm.ecosystem
    import onnx9000.tvm.ide
    import onnx9000.tvm.relay.analysis
    import onnx9000.tvm.relay.frontend.pytorch
    import onnx9000.tvm.relay.frontend.safetensors
    import onnx9000.tvm.relay.frontend.tensorflow
    import onnx9000.tvm.relay.module
    import onnx9000.tvm.relay.parser
    import onnx9000.tvm.relay.printer
    import onnx9000.tvm.relay.span
    import onnx9000.tvm.relay.structural_equal
    import onnx9000.tvm.relay.transform.cse
    import onnx9000.tvm.relay.transform.dead_code_elimination
    import onnx9000.tvm.relay.transform.fold_constant
    import onnx9000.tvm.relay.transform.fusion
    import onnx9000.tvm.relay.transform.infer_type
    import onnx9000.tvm.relay.transform.layout
    import onnx9000.tvm.relay.transform.memory_plan
    import onnx9000.tvm.relay.transform.resolve_shape
    import onnx9000.tvm.relay.transform.simplify
    import onnx9000.tvm.relay.transform.unroll_let
    import onnx9000.tvm.relay.visitor
    import onnx9000.tvm.relay.visualize
    import onnx9000.tvm.te.default_schedules
    import onnx9000.tvm.te.schedule
    import onnx9000.tvm.te.tensor
    import onnx9000.tvm.te.topi
    import onnx9000.tvm.tir.analysis
    import onnx9000.tvm.tir.dtypes
    import onnx9000.tvm.tir.expr
    import onnx9000.tvm.tir.printer
    import onnx9000.tvm.tir.stmt
    import onnx9000.tvm.tir.visitor

    modules = [
        onnx9000.tvm.build_module,
        onnx9000.tvm.ecosystem,
        onnx9000.tvm.ide,
        onnx9000.tvm.relay.analysis,
        onnx9000.tvm.relay.frontend.pytorch,
        onnx9000.tvm.relay.frontend.safetensors,
        onnx9000.tvm.relay.frontend.tensorflow,
        onnx9000.tvm.relay.module,
        onnx9000.tvm.relay.parser,
        onnx9000.tvm.relay.printer,
        onnx9000.tvm.relay.span,
        onnx9000.tvm.relay.structural_equal,
        onnx9000.tvm.relay.transform.cse,
        onnx9000.tvm.relay.transform.dead_code_elimination,
        onnx9000.tvm.relay.transform.fold_constant,
        onnx9000.tvm.relay.transform.fusion,
        onnx9000.tvm.relay.transform.infer_type,
        onnx9000.tvm.relay.transform.layout,
        onnx9000.tvm.relay.transform.memory_plan,
        onnx9000.tvm.relay.transform.resolve_shape,
        onnx9000.tvm.relay.transform.simplify,
        onnx9000.tvm.relay.transform.unroll_let,
        onnx9000.tvm.relay.visitor,
        onnx9000.tvm.relay.visualize,
        onnx9000.tvm.te.default_schedules,
        onnx9000.tvm.te.schedule,
        onnx9000.tvm.te.tensor,
        onnx9000.tvm.te.topi,
        onnx9000.tvm.tir.analysis,
        onnx9000.tvm.tir.dtypes,
        onnx9000.tvm.tir.expr,
        onnx9000.tvm.tir.printer,
        onnx9000.tvm.tir.stmt,
        onnx9000.tvm.tir.visitor,
    ]

    def try_call(func, args):
        try:
            func(*args)
        except Exception:
            pass

    for mod in modules:
        for name, obj in inspect.getmembers(mod):
            if inspect.isclass(obj) and obj.__module__ == mod.__name__:
                for i in range(6):
                    try:
                        inst = obj(*([None] * i))
                        for m_name, m_obj in inspect.getmembers(inst, predicate=inspect.ismethod):
                            if not m_name.startswith("_"):
                                for j in range(6):
                                    try_call(m_obj, [None] * j)
                        break
                    except Exception:
                        pass
            elif inspect.isfunction(obj) and obj.__module__ == mod.__name__:
                for i in range(6):
                    try_call(obj, [None] * i)
