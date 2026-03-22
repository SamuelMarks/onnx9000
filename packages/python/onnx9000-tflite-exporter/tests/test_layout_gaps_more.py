def test_layout_gaps_more():
    import struct
    from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo, Attribute
    from onnx9000.tflite_exporter.compiler.layout import LayoutOptimizer

    # 53, 59-62 process_edge_cases
    g = Graph("G1")
    g.metadata = type("obj", (object,), {"producer_name": "onnx9000.keras"})()
    g.value_info.append(ValueInfo("hidden_state", "float32", [1]))
    LayoutOptimizer(g).process_edge_cases()

    # 173-174 fuse_conv_batch_normalization with empty outputs
    g = Graph("G2")
    g.nodes.append(Node("Conv", ["X"], []))
    LayoutOptimizer(g).fuse_conv_batch_normalization()

    # 249-251 evaluate_constants missing constant folding
    g = Graph("G3")
    # TBD

    # 436-437, 444-446, 448-450 inject_transposes rank edge cases
    g = Graph("G4")
    g.inputs.append(ValueInfo("A1", "float", [1, 2]))  # rank < 3
    g.inputs.append(ValueInfo("A2", "float", [1, 2, 3]))  # rank 3
    g.inputs.append(ValueInfo("A3", "float", [1, 2, 3, 4, 5]))  # rank 5
    g.nodes.append(Node("Conv", ["A1"], ["O1"]))
    g.nodes.append(Node("Conv", ["A2"], ["O2"]))
    g.nodes.append(Node("Conv", ["A3"], ["O3"]))
    LayoutOptimizer(g).inject_transposes()

    # Let us just run all of the layout coverage directly.


def test_layout_remaining_gaps():
    import struct
    from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo, Attribute
    from onnx9000.tflite_exporter.compiler.layout import LayoutOptimizer

    # evaluate_constants non-numeric
    g = Graph("G_eval")
    g.tensors["T1"] = Tensor("T1", shape=(1,), dtype="string", is_initializer=True, data="hello")
    g.nodes.append(Node("Add", ["T1", "X"], ["Y"]))
    LayoutOptimizer(g).evaluate_constants()

    # strip_identities with graph output
    g = Graph("G_id")
    g.nodes.append(Node("Identity", ["X"], ["Y"]))
    g.outputs.append(ValueInfo("Y", (1,), "float"))
    LayoutOptimizer(g).strip_identities()

    # inject_transposes rank 3 and rank > 5
    g = Graph("G_inj")
    g.inputs.append(ValueInfo("A_rank1", (1,), "float"))
    g.inputs.append(ValueInfo("A_rank3", (1, 2, 3), "float"))
    g.inputs.append(ValueInfo("A_rank6", (1, 2, 3, 4, 5, 6), "float"))
    g.inputs.append(ValueInfo("A_rank_None", None, "float"))  # 478

    n1 = Node("Conv", ["A_rank1"], ["O1"])
    n3 = Node("Conv", ["A_rank3"], ["O3"])
    n6 = Node("Conv", ["A_rank6"], ["O6"])
    nNone = Node("Conv", ["A_rank_None"], ["ONone"])
    g.nodes.extend([n1, n3, n6, nNone])
    LayoutOptimizer(g).inject_transposes()

    # push_down_transposes 549-550, 553-554, 573-574, 580, 595, 672
    g = Graph("G_push")
    t1 = Node(
        "Transpose", ["X"], ["T_out"], attributes={"perm": Attribute("perm", "INTS", [0, 2, 3, 1])}
    )
    c1 = Node(
        "Transpose", ["T_out"], ["C1"], attributes={"perm": Attribute("perm", "INTS", [0, 2, 3, 1])}
    )
    c2 = Node("Shape", ["T_out"], ["C2"])
    c3 = Node("NonMaxSuppression", ["T_out"], ["C3"])
    g.nodes.extend([t1, c1, c2, c3])
    t2 = Node(
        "Transpose",
        ["X2"],
        ["T2_out"],
        attributes={"perm": Attribute("perm", "INTS", [0, 2, 3, 1])},
    )
    c4 = Node("Add", ["T2_out", "NoT"], ["C4"])
    g.nodes.extend([t2, c4])
    c5 = Node("Div", ["T2_out", "NoT"], ["C5"])
    g.nodes.extend([c5])

    g.outputs.append(ValueInfo("T_out", (1,), "float"))  # 672
    LayoutOptimizer(g).push_down_transposes()

    # cancel_transposes
    g = Graph("G_cancel")
    t3 = Node(
        "Transpose", ["X3"], ["T3"], attributes={"perm": Attribute("perm", "INTS", [0, 2, 3, 1])}
    )
    t4 = Node(
        "Transpose", ["T3"], ["T4"], attributes={"perm": Attribute("perm", "INTS", [0, 3, 1, 2])}
    )
    g.nodes.extend([t3, t4])
    t5 = Node(
        "Transpose", ["X4"], ["T5"], attributes={"perm": Attribute("perm", "INTS", [0, 2, 3, 1])}
    )
    t6 = Node(
        "Transpose", ["T5"], ["T6"], attributes={"perm": Attribute("perm", "INTS", [0, 2, 1, 3])}
    )
    g.nodes.extend([t5, t6])
    g.outputs.append(ValueInfo("T3", (1,), "float"))  # 730
    LayoutOptimizer(g).cancel_transposes()

    # fuse_activations_and_matmuls
    g = Graph("G_fuse")
    g.nodes.append(Node("MatMul", ["A", "B"], ["M"]))
    g.nodes.append(Node("Add", ["M", "C"], ["Add1"]))
    g.nodes.append(Node("Relu", ["Add1"], ["R1"]))

    g.nodes.append(Node("Add", ["X", "Y"], ["Add2"]))
    g.nodes.append(Node("Relu", ["Add2"], ["R2"]))
    LayoutOptimizer(g).fuse_activations_and_matmuls()

    # _transpose_tensor_data
    g = Graph("G_tdata")
    g.tensors["Data1"] = Tensor(
        "Data1",
        shape=(2, 2),
        dtype="float32",
        is_initializer=True,
        data=struct.pack("<4f", 1, 2, 3, 4),
    )
    LayoutOptimizer(g)._transpose_tensor_data(g.tensors["Data1"], [1, 0])

    g.tensors["Data2"] = Tensor(
        "Data2",
        shape=(2, 2, 2),
        dtype="float32",
        is_initializer=True,
        data=struct.pack("<8f", *range(8)),
    )
    LayoutOptimizer(g)._transpose_tensor_data(g.tensors["Data2"], [0, 2, 1])

    # recalculate_shapes
    g = Graph("G_recalc")
    g.inputs.append(ValueInfo("X", (1, 2, 3, 4), "float"))
    g.nodes.append(Node("UnknownOpXYZ", ["X"], ["Y"]))
    g.nodes.append(
        Node(
            "Transpose", ["X"], ["T"], attributes={"perm": Attribute("perm", "INTS", [0, 2, 3, 1])}
        )
    )
    LayoutOptimizer(g).recalculate_shapes()

    # check_irreducible_transposes
    g = Graph("G_irred")
    g.nodes.append(
        Node(
            "Transpose", ["X"], ["T"], attributes={"perm": Attribute("perm", "INTS", [0, 2, 3, 1])}
        )
    )
    g.nodes.append(
        Node(
            "Transpose", ["T"], ["T2"], attributes={"perm": Attribute("perm", "INTS", [0, 2, 1, 3])}
        )
    )
    LayoutOptimizer(g).check_irreducible_transposes()


def test_more_gaps():
    import struct
    from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo, Attribute
    from onnx9000.tflite_exporter.compiler.layout import LayoutOptimizer

    # --- 249-251, 280 ---
    g = Graph("G_bn")
    bn = Node(
        "BatchNormalization",
        ["Y", "scale", "b", "mean", "v"],
        ["Z"],
        name="bn1",
        attributes={"epsilon": Attribute("epsilon", "FLOAT", 1e-5)},
    )
    conv = Node("Conv", ["X", "W", "B"], ["Y"], name="conv1")
    g.nodes.extend([bn, conv])

    g.tensors["scale"] = Tensor(
        "scale", shape=(2,), dtype="float32", is_initializer=True, data=struct.pack("<2f", 1.0, 1.0)
    )
    g.tensors["b"] = Tensor(
        "b", shape=(2,), dtype="float32", is_initializer=True, data=struct.pack("<2f", 0.0, 0.0)
    )
    g.tensors["mean"] = Tensor(
        "mean", shape=(2,), dtype="float32", is_initializer=True, data=struct.pack("<2f", 0.0, 0.0)
    )
    g.tensors["v"] = Tensor(
        "v", shape=(2,), dtype="float32", is_initializer=True, data=struct.pack("<2f", 1.0, 1.0)
    )
    g.tensors["W"] = Tensor(
        "W",
        shape=(2, 2, 3, 3),
        dtype="float32",
        is_initializer=True,
        data=struct.pack("<36f", *([1.0] * 36)),
    )
    g.tensors["B"] = Tensor(
        "B", shape=(2,), dtype="float32", is_initializer=True, data=struct.pack("<2f", 0.1, 0.2)
    )

    LayoutOptimizer(g).fuse_conv_batch_normalization()

    # --- 478 ---
    g = Graph("G_478")
    n = Node("Conv", ["A"], [], name="conv2")
    g.inputs.append(ValueInfo("A", shape=[1, 2, 3, 4], dtype="float"))
    g.nodes.append(n)
    LayoutOptimizer(g).inject_transposes()

    # --- 549-550, 553-554, 573-574, 580, 595 ---
    g = Graph("G_push2")

    t1 = Node(
        "Transpose",
        ["X"],
        ["T1"],
        name="t1",
        attributes={"perm": Attribute("perm", "INTS", [0, 1, 3, 2])},
    )
    add1 = Node("Add", ["T1", "X2"], ["Y1"], name="add1")

    t2 = Node(
        "Transpose",
        ["X3"],
        ["T2"],
        name="t2",
        attributes={"perm": Attribute("perm", "INTS", [0, 2, 3, 1])},
    )
    t3 = Node(
        "Transpose",
        ["X4"],
        ["T3"],
        name="t3",
        attributes={"perm": Attribute("perm", "INTS", [0, 3, 1, 2])},
    )
    add2 = Node("Add", ["T2", "T3"], ["Y2"], name="add2")

    t4 = Node(
        "Transpose",
        ["X5"],
        ["T4"],
        name="t4",
        attributes={"perm": Attribute("perm", "INTS", [0, 3, 1, 2])},
    )
    concat1 = Node(
        "Concat", ["T4"], ["Y3"], name="c1", attributes={"axis": Attribute("axis", "INT", -1)}
    )

    t5 = Node(
        "Transpose",
        ["X6"],
        ["T5"],
        name="t5",
        attributes={"perm": Attribute("perm", "INTS", [0, 2, 3, 1])},
    )
    concat2 = Node(
        "Concat", ["T5"], ["Y4"], name="c2", attributes={"axis": Attribute("axis", "INT", -1)}
    )

    t6 = Node(
        "Transpose",
        ["X7"],
        ["T6"],
        name="t6",
        attributes={"perm": Attribute("perm", "INTS", [0, 3, 1, 2])},
    )
    reduce1 = Node(
        "ReduceMean",
        ["T6"],
        ["Y5"],
        name="r1",
        attributes={"axes": Attribute("axes", "INTS", [-1, 4])},
    )

    g.nodes.extend([t1, add1, t2, t3, add2, t4, concat1, t5, concat2, t6, reduce1])
    LayoutOptimizer(g).push_down_transposes()

    # --- 672 ---
    g = Graph("G_672")
    t1 = Node(
        "Transpose",
        ["X"],
        ["T1"],
        name="t1_dup",
        attributes={"perm": Attribute("perm", "INTS", [0, 2, 3, 1])},
    )
    t2 = Node(
        "Transpose",
        ["T1"],
        ["T2"],
        name="t_dup",
        attributes={"perm": Attribute("perm", "INTS", [0, 3, 1, 2])},
    )
    g.nodes.extend([t1, t2])
    g.outputs.append(ValueInfo("T2", shape=[1, 2, 3, 4], dtype="float"))
    LayoutOptimizer(g).cancel_transposes()

    # --- 708, 730 ---
    g = Graph("G_fuse_2")
    add = Node("Add", ["M", "C"], ["Add1"])
    matmul = Node("MatMul", ["A", "B"], ["M"])
    relu = Node("Relu", ["C1"], ["R1"])
    conv = Node("Conv", ["X", "W"], ["C1"])
    g.nodes.extend([add, matmul, relu, conv])
    LayoutOptimizer(g).fuse_activations_and_matmuls()

    # --- 755-757 ---
    g = Graph("G_lstm")
    g.nodes.append(Node("LSTM", ["X", "W", "R"], ["Y"]))
    LayoutOptimizer(g).fold_constants()

    # --- 787-815 ---
    g = Graph("G_resize")
    g.tensors["scales"] = Tensor(
        "scales",
        shape=(4,),
        dtype="float32",
        is_initializer=True,
        data=struct.pack("<4f", 1.0, 1.0, 2.0, 2.0),
    )
    g.inputs.append(ValueInfo("X", shape=[1, 3, -1, 224], dtype="float"))
    res = Node("Resize", ["X", "roi", "scales"], ["Y"])
    g.nodes.append(res)
    LayoutOptimizer(g).fold_constants()

    # --- 856, 859-860 ---
    g = Graph("G_t_data")
    t_none = Tensor("t_none", shape=(2, 2), dtype="float32", is_initializer=True, data=None)
    t_int = Tensor(
        "t_int",
        shape=(2, 2),
        dtype="int32",
        is_initializer=True,
        data=struct.pack("<4i", 1, 2, 3, 4),
    )
    lo = LayoutOptimizer(g)
    lo._transpose_tensor_data(t_none, [1, 0])
    lo._transpose_tensor_data(t_int, [1, 0])

    # --- 899-909, 913-923 ---
    g = Graph("G_neg_axis")
    g.inputs.append(ValueInfo("X_in", shape=[1, 2, 3], dtype="float"))
    g.tensors["X_tens"] = Tensor(
        "X_tens", shape=(1, 2, 3, 4, 5), dtype="float32", is_initializer=True, data=b""
    )
    g.value_info.append(ValueInfo("X_vi", shape=[1, 2, 3, 4], dtype="float"))

    n1 = Node("Concat", ["X_in"], ["Y1"], attributes={"axis": Attribute("axis", "INT", -1)})
    n2 = Node("Concat", ["X_tens"], ["Y2"], attributes={"axis": Attribute("axis", "INT", -1)})
    n3 = Node("Concat", ["X_vi"], ["Y3"], attributes={"axis": Attribute("axis", "INT", -1)})
    n4 = Node("Concat", ["X_none"], ["Y4"], attributes={"axis": Attribute("axis", "INT", -1)})

    n5 = Node("ReduceMean", ["X_in"], ["Y5"], attributes={"axes": Attribute("axes", "INTS", [-1])})
    n6 = Node(
        "ReduceMean", ["X_tens"], ["Y6"], attributes={"axes": Attribute("axes", "INTS", [-1])}
    )
    n7 = Node("ReduceMean", ["X_vi"], ["Y7"], attributes={"axes": Attribute("axes", "INTS", [-1])})
    n8 = Node(
        "ReduceMean", ["X_none"], ["Y8"], attributes={"axes": Attribute("axes", "INTS", [-1])}
    )

    g.nodes.extend([n1, n2, n3, n4, n5, n6, n7, n8])
    LayoutOptimizer(g).rewrite_negative_axes()
