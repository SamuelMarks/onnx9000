"""Module providing core logic and structural definitions for the Core IR."""

import ctypes
import logging
from typing import Any, Optional, Union
from onnx9000.core.dtypes import DType

logger = logging.getLogger(__name__)


class DLDataType(ctypes.Structure):
    """DLPack data type structure."""

    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class DLDevice(ctypes.Structure):
    """DLPack device structure."""

    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class DLTensor(ctypes.Structure):
    """DLPack tensor structure."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class DLManagedTensor(ctypes.Structure):
    """DLPack managed tensor structure."""


DLManagedTensor._fields_ = [
    ("dl_tensor", DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensor))),
]


class DynamicDim:
    """Class DynamicDim implementation."""

    def __init__(self, value: Union[str, int]) -> None:
        """Initialize the class with necessary attributes."""
        self.value = value

    def __repr__(self) -> str:
        return f"DynamicDim({self.value})"

    def __str__(self) -> str:
        return str(self.value)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, DynamicDim):
            return self.value == other.value
        return False


class Attribute:
    """Class Attribute implementation."""

    @staticmethod
    def infer_type(value: Any) -> str:
        """Infer Type function logic implementation."""
        if isinstance(value, int):
            return "INT"
        if isinstance(value, float):
            return "FLOAT"
        if isinstance(value, str):
            return "STRING"
        if isinstance(value, Tensor):
            return "TENSOR"
        if isinstance(value, Graph):
            return "GRAPH"
        if isinstance(value, list):
            if not value:
                return "INTS"
            if isinstance(value[0], int):
                return "INTS"
            if isinstance(value[0], float):
                return "FLOATS"
            if isinstance(value[0], str):
                return "STRINGS"
            if isinstance(value[0], Tensor):
                return "TENSORS"
            if isinstance(value[0], Graph):
                return "GRAPHS"
        return "UNKNOWN"

    def __init__(self, name: str, attr_type: Optional[str] = None, value: Any = None) -> None:
        """Initialize the class with necessary attributes."""
        self.name = name
        self.value = value
        self.attr_type = attr_type if attr_type is not None else Attribute.infer_type(value)

    def __repr__(self) -> str:
        return f"Attribute(name={self.name}, type={self.attr_type}, value={self.value})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Attribute):
            return False
        return (
            self.name == other.name
            and self.attr_type == other.attr_type
            and (self.value == other.value)
        )


class ValueInfo:
    """Class ValueInfo implementation."""

    def __init__(self, name: str, shape: tuple[Union[int, DynamicDim], ...], dtype: DType) -> None:
        """Initialize the class with necessary attributes."""
        self.name = name
        self.shape = shape
        self.dtype = dtype

    def __repr__(self) -> str:
        return f"ValueInfo(name={self.name}, shape={self.shape}, dtype={self.dtype})"


class Tensor:
    """Base Internal Representation of a Tensor."""

    def __init__(
        self, name: str, shape=None, dtype=None, is_initializer=False, requires_grad=True, data=None
    ) -> None:
        """Initialize the class with necessary attributes."""
        self.name = name
        self.inputs: list[Node] = []
        self.outputs: list[Node] = []
        self.shape: tuple[Union[int, DynamicDim], ...] = shape or tuple()
        self.dtype: Optional[DType] = dtype
        self.is_initializer: bool = is_initializer
        self.requires_grad: bool = requires_grad
        self.data: Optional[Union[bytes, memoryview, bytearray]] = data
        self.buffer_id: Optional[int] = None
        self.lifespan: tuple[int, int] = (-1, -1)

    def __repr__(self) -> str:
        return f"ir.Tensor(name={self.name})"

    def __hash__(self) -> int:
        return object.__hash__(self)

    def copy(self) -> "Tensor":
        """Copy function logic implementation."""
        if isinstance(self, Constant):
            return Constant(self.name, values=self.data, shape=self.shape, dtype=self.dtype)
        else:
            return Variable(self.name, shape=self.shape, dtype=self.dtype)

    def clear_inputs(self) -> None:
        """Disconnect from producer."""
        for n in self.inputs:
            n.outputs = [o for o in n.outputs if o is not self]
        self.inputs.clear()

    def clear_outputs(self) -> None:
        """Disconnect from all consumers."""
        for n in self.outputs:
            n.inputs = [i for i in n.inputs if i is not self]
        self.outputs.clear()


class Variable(Tensor):
    """Internal Representation of a dynamic tensor."""

    def __init__(
        self,
        name: str,
        shape: Optional[tuple[Union[int, DynamicDim], ...]] = None,
        dtype: Optional[DType] = None,
    ) -> None:
        """Initialize the class with necessary attributes."""
        super().__init__(name)
        self.shape = shape or tuple()
        self.dtype = dtype

    def is_dynamic(self) -> bool:
        """Is Dynamic function logic implementation."""
        return any((isinstance(dim, DynamicDim) or dim == -1 for dim in self.shape))

    def is_empty(self) -> bool:
        """Is Empty function logic implementation."""
        return len(self.shape) == 0

    def __repr__(self) -> str:
        return f"ir.Variable(name={self.name}, shape={self.shape}, dtype={self.dtype})"


class Constant(Tensor):
    """Internal Representation of a static tensor."""

    def __init__(
        self,
        name: str,
        values: Optional[Union[bytes, memoryview, bytearray]] = None,
        shape: Optional[tuple[Union[int, DynamicDim], ...]] = None,
        dtype: Optional[DType] = None,
    ) -> None:
        """Initialize the class with necessary attributes."""
        super().__init__(name)
        self.data = values
        self.shape = shape or tuple()
        self.dtype = dtype
        self.is_initializer = True

    def __repr__(self) -> str:
        return f"ir.Constant(name={self.name}, shape={self.shape}, dtype={self.dtype})"

    @property
    def values(self):
        """Values function logic implementation."""
        return self.data

    @values.setter
    def values(self, val):
        """Values function logic implementation."""
        self.data = val

    def __dlpack_device__(self) -> tuple[int, int]:
        return (1, 0)

    def __dlpack__(self, stream: Optional[int] = None) -> Any:
        if self.data is None:
            raise ValueError("Cannot create DLPack capsule for a tensor with no data.")
        for dim in self.shape:
            if isinstance(dim, DynamicDim) or dim < 0:
                raise ValueError("DLPack export requires static shapes.")
        managed_tensor = DLManagedTensor()
        buffer_info = (ctypes.c_char * len(self.data)).from_buffer_copy(self.data)
        managed_tensor.dl_tensor.data = ctypes.cast(buffer_info, ctypes.c_void_p)
        managed_tensor.dl_tensor.device.device_type = 1
        managed_tensor.dl_tensor.device.device_id = 0
        managed_tensor.dl_tensor.ndim = len(self.shape)
        managed_tensor.dl_tensor.dtype.code = 2
        managed_tensor.dl_tensor.dtype.bits = 32
        managed_tensor.dl_tensor.dtype.lanes = 1
        shape_array = (ctypes.c_int64 * len(self.shape))(*self.shape)
        managed_tensor.dl_tensor.shape = ctypes.cast(shape_array, ctypes.POINTER(ctypes.c_int64))
        strides_array = (ctypes.c_int64 * len(self.shape))()
        stride = 1
        for i in reversed(range(len(self.shape))):
            strides_array[i] = stride
            stride *= int(self.shape[i])
        managed_tensor.dl_tensor.strides = ctypes.cast(
            strides_array, ctypes.POINTER(ctypes.c_int64)
        )
        managed_tensor.dl_tensor.byte_offset = 0
        PyCapsule_New = ctypes.pythonapi.PyCapsule_New
        PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        PyCapsule_New.restype = ctypes.py_object
        name_c = b"dltensor"
        ptr = ctypes.pointer(managed_tensor)
        capsule = PyCapsule_New(ptr, name_c, None)
        return capsule


class Node:
    """Internal Representation of an operation."""

    def __init__(
        self,
        op_type: str,
        inputs: Optional[list[Union[str, Tensor]]] = None,
        outputs: Optional[list[Union[str, Tensor]]] = None,
        attributes: Optional[dict[str, Attribute]] = None,
        name: str = "",
        domain: str = "",
    ) -> None:
        """Initialize the class with necessary attributes."""
        self.op_type = op_type
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.attributes = attributes or {}
        self.name = name
        self.domain = domain
        for i in self.inputs:
            if isinstance(i, Tensor) and self not in i.outputs:
                i.outputs.append(self)
        for o in self.outputs:
            if isinstance(o, Tensor) and self not in o.inputs:
                o.inputs.append(self)

    @property
    def op(self) -> str:
        """Op function logic implementation."""
        return self.op_type

    @op.setter
    def op(self, value: str) -> None:
        """Op function logic implementation."""
        self.op_type = value

    @property
    def attrs(self) -> dict[str, Attribute]:
        """Attrs function logic implementation."""
        return self.attributes

    def __hash__(self) -> int:
        return object.__hash__(self)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Node):
            return False
        if self.op_type != other.op_type:
            return False
        if len(self.inputs) != len(other.inputs):
            return False
        if len(self.outputs) != len(other.outputs):
            return False
        if self.attributes != other.attributes:
            return False
        in_names1 = [i.name if isinstance(i, Tensor) else i for i in self.inputs]
        in_names2 = [i.name if isinstance(i, Tensor) else i for i in other.inputs]
        out_names1 = [o.name if isinstance(o, Tensor) else o for o in self.outputs]
        out_names2 = [o.name if isinstance(o, Tensor) else o for o in other.outputs]
        return in_names1 == in_names2 and out_names1 == out_names2

    def copy(self, tensor_map=None) -> "Node":
        """Copy function logic implementation."""
        tensor_map = tensor_map or {}
        new_inputs = [
            tensor_map.get(i.name, i) if isinstance(i, Tensor) else i for i in self.inputs
        ]
        new_outputs = [
            tensor_map.get(o.name, o) if isinstance(o, Tensor) else o for o in self.outputs
        ]
        new_attrs = {
            k: Attribute(v.name, v.attr_type, v.value) for (k, v) in self.attributes.items()
        }
        return Node(
            op_type=self.op_type,
            inputs=new_inputs,
            outputs=new_outputs,
            attributes=new_attrs,
            name=self.name,
            domain=self.domain,
        )

    def i(self, idx: int = 0) -> Any:
        """Utility for quick input tensor retrieval."""
        return self.inputs[idx]

    def o(self, idx: int = 0) -> Any:
        """Utility for quick output tensor retrieval."""
        return self.outputs[idx]

    def __repr__(self) -> str:
        in_str = [i.name if isinstance(i, Tensor) else str(i) for i in self.inputs]
        out_str = [o.name if isinstance(o, Tensor) else str(o) for o in self.outputs]
        return f"ir.Node({self.op_type}, {in_str} -> {out_str})"


class Graph:
    """Internal Representation of a complete topological execution plan."""

    def __init__(self, name: str) -> None:
        """Initialize the class with necessary attributes."""
        self.name = name
        self.nodes: list[Node] = []
        self.tensors: dict[str, Tensor] = {}
        self._string_pool: dict[str, str] = {}
        self._node_name_counter: dict[str, int] = {}
        self._tensor_name_counter: dict[str, int] = {}
        self.inputs: list[ValueInfo] = []
        self.outputs: list[ValueInfo] = []
        self.initializers: list[str] = []
        self.opset_imports: dict[str, int] = {}
        self.doc_string: str = ""
        self.producer_map: dict[str, Node] = {}
        self.consumer_map: dict[str, list[Node]] = {}

    def _intern_string(self, s: str) -> str:
        if s not in self._string_pool:
            self._string_pool[s] = s
        return self._string_pool[s]

    def _uniquify_node_name(self, base_name: str) -> str:
        if not base_name:
            base_name = "node"
        if base_name not in self._node_name_counter:
            self._node_name_counter[base_name] = 0
            if not any((n.name == base_name for n in self.nodes)):
                return base_name
        while True:
            self._node_name_counter[base_name] += 1
            new_name = f"{base_name}_{self._node_name_counter[base_name]}"
            if not any((n.name == new_name for n in self.nodes)):
                return new_name

    def _uniquify_tensor_name(self, base_name: str) -> str:
        if not base_name:
            base_name = "tensor"
        if base_name not in self._tensor_name_counter:
            self._tensor_name_counter[base_name] = 0
            if base_name not in self.tensors:
                return base_name
        while True:
            self._tensor_name_counter[base_name] += 1
            new_name = f"{base_name}_{self._tensor_name_counter[base_name]}"
            if new_name not in self.tensors:
                return new_name

    def add_tensor(self, tensor: Tensor) -> None:
        """Add Tensor function logic implementation."""
        if tensor.name in self.tensors and self.tensors[tensor.name] is not tensor:
            tensor.name = self._uniquify_tensor_name(tensor.name)
        tensor.name = self._intern_string(tensor.name)
        self.tensors[tensor.name] = tensor

    def add_node(self, node: Node) -> None:
        """Add Node function logic implementation."""
        if any((n.name == node.name for n in self.nodes)) and node.name:
            node.name = self._uniquify_node_name(node.name)
        elif not node.name:
            node.name = self._uniquify_node_name(node.op_type)
        node.name = self._intern_string(node.name)
        self.nodes.append(node)
        for o in node.outputs:
            if isinstance(o, Tensor):
                self.producer_map[o.name] = node
        for i in node.inputs:
            if isinstance(i, Tensor):
                if i.name not in self.consumer_map:
                    self.consumer_map[i.name] = []
                self.consumer_map[i.name].append(node)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Graph):
            return False
        if self.name != other.name:
            return False
        if len(self.nodes) != len(other.nodes):
            return False
        for n1, n2 in zip(self.nodes, other.nodes):
            if n1 != n2:
                return False
        return True

    def copy(self) -> "Graph":
        """Copy function logic implementation."""
        g = Graph(self.name)
        g.doc_string = self.doc_string
        g.opset_imports = self.opset_imports.copy()
        tensor_map = {}
        for t_name, t in self.tensors.items():
            new_t = t.copy()
            tensor_map[t_name] = new_t
            g.add_tensor(new_t)
        for n in self.nodes:
            g.add_node(n.copy(tensor_map))
        g.inputs = [ValueInfo(v.name, v.shape, v.dtype) for v in self.inputs]
        g.outputs = [ValueInfo(v.name, v.shape, v.dtype) for v in self.outputs]
        g.initializers = self.initializers.copy()
        return g

    def tensors(self):
        """Lazy evaluation dictionary generator or returns the dict."""
        return self.tensors

    def get_node(self, name: str) -> Optional[Node]:
        """Get Node function logic implementation."""
        for n in self.nodes:
            if n.name == name:
                return n
        return None

    def print_visualizer(self) -> None:
        """Print Visualizer function logic implementation."""
        logger.info(f"=== Graph: {self.name} ===")
        in_names = [i.name for i in self.inputs]
        out_names = [o.name for o in self.outputs]
        logger.info(f"Inputs: {in_names}")
        logger.info(f"Outputs: {out_names}")
        logger.info("Nodes:")
        for idx, node in enumerate(self.nodes):
            in_str = [i.name if isinstance(i, Tensor) else str(i) for i in node.inputs]
            out_str = [o.name if isinstance(o, Tensor) else str(o) for o in node.outputs]
            logger.info(f"  [{idx}] {node.op_type}: {in_str} -> {out_str}")
        logger.info("=========================")
