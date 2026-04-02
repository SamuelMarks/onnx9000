"""Module providing core logic and structural definitions for the Core IR."""

import ctypes
import logging
from typing import Any, Optional, Union

from onnx9000.core.dtypes import DType

logger = logging.getLogger(__name__)


class DLDataType(ctypes.Structure):
    """DLPack data type structure."""

    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


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
    """Represents a dimension that can be either a symbolic string or an integer."""

    def __init__(self, value: Union[str, int]) -> None:
        """Initialize a DynamicDim.

        Args:
            value: The symbolic name or integer value of the dimension.
        """
        self.value = value

    def __repr__(self) -> str:
        """Return a string representation of the DynamicDim."""
        return f"DynamicDim({self.value})"

    def __str__(self) -> str:
        """Return the string value of the dimension."""
        return str(self.value)

    def __eq__(self, other: Any) -> bool:
        """Check equality with another DynamicDim or value."""
        if isinstance(other, DynamicDim):
            return self.value == other.value
        return False


class Attribute:
    """Represents an attribute of a Node."""

    @staticmethod
    def infer_type(value: Any) -> str:
        """Infer the ONNX attribute type from a Python value.

        Args:
            value: The Python value to infer the type for.

        Returns:
            A string representing the inferred ONNX attribute type.
        """
        if isinstance(value, int):
            return "INT"
        if isinstance(value, float):
            return "FLOAT"
        if isinstance(value, str):
            return "STRING"
        if isinstance(value, SparseTensor):
            return "SPARSE_TENSOR"
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
            if isinstance(value[0], SparseTensor):
                return "SPARSE_TENSORS"
            if isinstance(value[0], Tensor):
                return "TENSORS"
            if isinstance(value[0], Graph):
                return "GRAPHS"
        return "UNKNOWN"

    def __init__(self, name: str, attr_type: Optional[str] = None, value: Any = None) -> None:
        """Initialize an Attribute.

        Args:
            name: The name of the attribute.
            attr_type: The ONNX type of the attribute. If None, it is inferred.
            value: The value of the attribute.
        """
        self.name = name
        self.value = value
        self.attr_type = attr_type if attr_type is not None else Attribute.infer_type(value)

    def __repr__(self) -> str:
        """Return a string representation of the Attribute."""
        return f"Attribute(name={self.name}, type={self.attr_type}, value={self.value})"

    def __eq__(self, other: Any) -> bool:
        """Check equality with another Attribute."""
        if not isinstance(other, Attribute):
            return False
        return (
            self.name == other.name
            and self.attr_type == other.attr_type
            and (self.value == other.value)
        )


class ValueInfo:
    """Represents metadata about a Tensor (name, shape, and type)."""

    def __init__(self, name: str, shape: tuple[Union[int, DynamicDim], ...], dtype: DType) -> None:
        """Initialize ValueInfo.

        Args:
            name: The name of the value.
            shape: The shape of the tensor.
            dtype: The data type of the tensor.
        """
        self.name = name
        self.shape = shape
        self.dtype = dtype

    def __repr__(self) -> str:
        """Return a string representation of the ValueInfo."""
        return f"ValueInfo(name={self.name}, shape={self.shape}, dtype={self.dtype})"


class Tensor:
    """Base Internal Representation of a Tensor."""

    def __init__(
        self,
        name: str,
        shape=None,
        dtype=None,
        is_initializer=False,
        requires_grad=True,
        data=None,
    ) -> None:
        """Initialize a Tensor.

        Args:
            name: Unique name of the tensor.
            shape: Shape tuple, can contain DynamicDim objects.
            dtype: DType of the tensor.
            is_initializer: Whether this tensor is a constant initializer.
            requires_grad: Whether this tensor requires gradients.
            data: Raw data bytes if it is an initializer.
        """
        self.name = name
        self.inputs: list[Node] = []
        self.outputs: list[Node] = []
        self.shape: tuple[Union[int, DynamicDim], ...] = shape or ()
        self.dtype: Optional[DType] = dtype
        self.is_initializer: bool = is_initializer
        self.requires_grad: bool = requires_grad
        self.data: Optional[Union[bytes, memoryview, bytearray]] = data
        self.buffer_id: Optional[int] = None
        self.lifespan: tuple[int, int] = (-1, -1)

    def __repr__(self) -> str:
        """Return a string representation of the Tensor."""
        return f"ir.Tensor(name={self.name})"

    def __hash__(self) -> int:
        """Return the hash of the Tensor object."""
        return object.__hash__(self)

    def copy(self) -> "Tensor":
        """Return a deep copy of the Tensor.

        Returns:
            A new Tensor object with copied attributes.
        """
        if isinstance(self, Constant):
            return Constant(self.name, values=self.data, shape=self.shape, dtype=self.dtype)
        elif isinstance(self, SparseTensor):
            return SparseTensor(
                self.name,
                values=self.values.copy() if self.values else None,
                indices=self.indices.copy() if self.indices else None,
                dims=self.shape,
                format=self.format,
                row_ptr=self.row_ptr.copy() if self.row_ptr else None,
                col_indices=self.col_indices.copy() if self.col_indices else None,
                block_dims=self.block_dims,
            )
        else:
            return Variable(self.name, shape=self.shape, dtype=self.dtype)

    def clear_inputs(self) -> None:
        """Disconnect the tensor from its producer nodes."""
        for n in self.inputs:
            n.outputs = [o for o in n.outputs if o is not self]
        self.inputs.clear()

    def clear_outputs(self) -> None:
        """Disconnect the tensor from all its consumer nodes."""
        for n in self.outputs:
            n.inputs = [i for i in n.inputs if i is not self]
        self.outputs.clear()


class SparseTensor(Tensor):
    """Internal Representation of a Sparse Tensor."""

    def __init__(
        self,
        name: str,
        values: Optional["Constant"] = None,
        indices: Optional["Constant"] = None,
        dims: Optional[tuple[Union[int, DynamicDim], ...]] = None,
        format: str = "COO",
        row_ptr: Optional["Constant"] = None,
        col_indices: Optional["Constant"] = None,
        block_dims: Optional[tuple[int, ...]] = None,
    ) -> None:
        """Initialize a SparseTensor.

        Args:
            name: Unique name of the sparse tensor.
            values: Constant containing the non-zero values.
            indices: Constant containing the indices of non-zero values.
            dims: Original dense shape of the tensor.
            format: Sparse format (e.g., 'COO', 'CSR', 'CSC').
            row_ptr: Row pointers for CSR format.
            col_indices: Column indices for CSC format.
            block_dims: Block dimensions for blocked formats.
        """
        super().__init__(name)
        self.values = values
        self.indices = indices
        self.row_ptr = row_ptr
        self.col_indices = col_indices
        self.block_dims = block_dims
        self.shape = dims or ()
        self.format = format
        self.is_initializer = True

    def __repr__(self) -> str:
        """Return a string representation of the SparseTensor."""
        return f"ir.SparseTensor(name={self.name}, shape={self.shape}, format={self.format})"

    def copy(self) -> "SparseTensor":
        """Return a deep copy of the SparseTensor.

        Returns:
            A new SparseTensor object.
        """
        return SparseTensor(
            self.name,
            values=self.values.copy() if self.values else None,
            indices=self.indices.copy() if self.indices else None,
            dims=self.shape,
            format=self.format,
            row_ptr=self.row_ptr.copy() if self.row_ptr else None,
            col_indices=self.col_indices.copy() if self.col_indices else None,
            block_dims=self.block_dims,
        )


class Variable(Tensor):
    """Internal Representation of a dynamic tensor whose values are computed at runtime."""

    def __init__(
        self,
        name: str,
        shape: Optional[tuple[Union[int, DynamicDim], ...]] = None,
        dtype: Optional[DType] = None,
    ) -> None:
        """Initialize a Variable tensor.

        Args:
            name: Unique name of the tensor.
            shape: Shape of the tensor.
            dtype: Data type of the tensor.
        """
        super().__init__(name)
        self.shape = shape or ()
        self.dtype = dtype

    def is_dynamic(self) -> bool:
        """Check if any dimension of the tensor is dynamic.

        Returns:
            True if any dimension is symbolic or unknown (-1).
        """
        return any(isinstance(dim, DynamicDim) or dim == -1 for dim in self.shape)

    def is_empty(self) -> bool:
        """Check if the tensor has an empty shape.

        Returns:
            True if the shape tuple is empty.
        """
        return len(self.shape) == 0

    def __repr__(self) -> str:
        """Return a string representation of the Variable."""
        return f"ir.Variable(name={self.name}, shape={self.shape}, dtype={self.dtype})"


class Constant(Tensor):
    """Internal Representation of a static tensor with fixed values."""

    def __init__(
        self,
        name: str,
        values: Optional[Union[bytes, memoryview, bytearray]] = None,
        shape: Optional[tuple[Union[int, DynamicDim], ...]] = None,
        dtype: Optional[DType] = None,
    ) -> None:
        """Initialize a Constant tensor.

        Args:
            name: Unique name of the tensor.
            values: Raw data bytes for the tensor.
            shape: Shape of the tensor.
            dtype: Data type of the tensor.
        """
        super().__init__(name)
        self.data = values
        self.shape = shape or ()
        self.dtype = dtype
        self.is_initializer = True

    def __repr__(self) -> str:
        """Return a string representation of the Constant."""
        return f"ir.Constant(name={self.name}, shape={self.shape}, dtype={self.dtype})"

    @property
    def values(self):
        """Get the raw data bytes of the constant."""
        return self.data

    @values.setter
    def values(self, val) -> None:
        """Set the raw data bytes of the constant."""
        self.data = val

    def __dlpack_device__(self) -> tuple[int, int]:
        """Return the DLPack device info (CPU)."""
        return (1, 0)

    def __dlpack__(self, stream: Optional[int] = None) -> Any:
        """Export the constant as a DLPack capsule.

        Args:
            stream: Optional stream for synchronization.

        Returns:
            A PyCapsule containing the DLManagedTensor.

        Raises:
            ValueError: If the tensor has no data or has dynamic shapes.
        """
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
    """Internal Representation of an operation node in the graph."""

    def __init__(
        self,
        op_type: str,
        inputs: Optional[list[Union[str, Tensor]]] = None,
        outputs: Optional[list[Union[str, Tensor]]] = None,
        attributes: Optional[dict[str, Attribute]] = None,
        name: str = "",
        domain: str = "",
    ) -> None:
        """Initialize an Operation Node.

        Args:
            op_type: The ONNX operator type (e.g., 'Add', 'Conv').
            inputs: List of input tensors or tensor names.
            outputs: List of output tensors or tensor names.
            attributes: Dictionary of node attributes.
            name: Unique name of the node.
            domain: The domain of the operator (default is ai.onnx).
        """
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
        """Get the operator type."""
        return self.op_type

    @op.setter
    def op(self, value: str) -> None:
        """Set the operator type."""
        self.op_type = value

    @property
    def attrs(self) -> dict[str, Attribute]:
        """Get the node attributes."""
        return self.attributes

    def __hash__(self) -> int:
        """Return the hash of the Node object."""
        return object.__hash__(self)

    def __eq__(self, other: Any) -> bool:
        """Check equality with another Node."""
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
        """Return a deep copy of the Node.

        Args:
            tensor_map: Mapping of old tensor names to new Tensor objects.

        Returns:
            A new Node object with copied attributes and linked tensors.
        """
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
        """Retrieve the input tensor at the specified index."""
        return self.inputs[idx]

    def o(self, idx: int = 0) -> Any:
        """Retrieve the output tensor at the specified index."""
        return self.outputs[idx]

    def __repr__(self) -> str:
        """Return a string representation of the Node."""
        in_str = [i.name if isinstance(i, Tensor) else str(i) for i in self.inputs]
        out_str = [o.name if isinstance(o, Tensor) else str(o) for o in self.outputs]
        return f"ir.Node({self.op_type}, {in_str} -> {out_str})"


class Graph:
    """Internal Representation of a complete topological execution plan."""

    def __init__(self, name: str) -> None:
        """Initialize a Graph.

        Args:
            name: The name of the graph.
        """
        self.name = name
        self.nodes: list[Node] = []
        self.tensors: dict[str, Tensor] = {}
        self._string_pool: dict[str, str] = {}
        self._node_name_counter: dict[str, int] = {}
        self._tensor_name_counter: dict[str, int] = {}
        self.inputs: list[ValueInfo] = []
        self.outputs: list[ValueInfo] = []
        self.value_info: list[ValueInfo] = []
        self.initializers: list[str] = []
        self.sparse_initializers: list[str] = []
        self.opset_imports: dict[str, int] = {}
        self.doc_string: str = ""
        self.producer_name: str = "onnx9000"
        self.producer_version: str = "1.0.0"
        self.metadata_props: dict[str, str] = {}
        self.producer_map: dict[str, Node] = {}
        self.consumer_map: dict[str, list[Node]] = {}

    def _intern_string(self, s: str) -> str:
        """Intern a string to save memory and ensure uniqueness.

        Args:
            s: The string to intern.

        Returns:
            The interned string.
        """
        if s not in self._string_pool:
            self._string_pool[s] = s
        return self._string_pool[s]

    def _uniquify_node_name(self, base_name: str) -> str:
        """Generate a unique name for a node.

        Args:
            base_name: The desired base name.

        Returns:
            A unique node name.
        """
        if not base_name:
            base_name = "node"
        if base_name not in self._node_name_counter:
            self._node_name_counter[base_name] = 0
            if not any(n.name == base_name for n in self.nodes):
                return base_name
        while True:
            self._node_name_counter[base_name] += 1
            new_name = f"{base_name}_{self._node_name_counter[base_name]}"
            if not any(n.name == new_name for n in self.nodes):
                return new_name

    def _uniquify_tensor_name(self, base_name: str) -> str:
        """Generate a unique name for a tensor.

        Args:
            base_name: The desired base name.

        Returns:
            A unique tensor name.
        """
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
        """Add a Tensor to the graph.

        Args:
            tensor: The Tensor object to add.
        """
        if tensor.name in self.tensors and self.tensors[tensor.name] is not tensor:
            tensor.name = self._uniquify_tensor_name(tensor.name)
        tensor.name = self._intern_string(tensor.name)
        self.tensors[tensor.name] = tensor

    def add_node(self, node: Node) -> None:
        """Add a Node to the graph and update connectivity maps.

        Ensures node name uniqueness, updates producer/consumer maps, and maintains
        tensor-node bi-directional links.

        Args:
            node: The Node object to add to the graph.
        """
        if any(n.name == node.name for n in self.nodes) and node.name:
            node.name = self._uniquify_node_name(node.name)
        elif not node.name:
            node.name = self._uniquify_node_name(node.op_type)
        node.name = self._intern_string(node.name)
        self.nodes.append(node)

        for o in node.outputs:
            o_name = o.name if isinstance(o, Tensor) else o
            self.producer_map[o_name] = node
            if isinstance(o, Tensor):
                if node not in o.inputs:
                    o.inputs.append(node)

        for i in node.inputs:
            i_name = i.name if isinstance(i, Tensor) else i
            if i_name not in self.consumer_map:
                self.consumer_map[i_name] = []
            if node not in self.consumer_map[i_name]:
                self.consumer_map[i_name].append(node)
            if isinstance(i, Tensor):
                if node not in i.outputs:
                    i.outputs.append(node)

    def disconnect_node(self, node: Node) -> None:
        """Disconnect a node from all its input and output tensors.

        Args:
            node: The Node object to disconnect.
        """
        for i in node.inputs:
            if isinstance(i, Tensor) and node in i.outputs:
                i.outputs.remove(node)
        for o in node.outputs:
            if isinstance(o, Tensor) and node in o.inputs:
                o.inputs.remove(node)

    def append_node(self, node: Node) -> None:
        """Alias for add_node.

        Args:
            node: The Node object to add.
        """
        self.add_node(node)

    def remove_node(self, node: Node) -> None:
        """Remove a Node from the graph and update connectivity maps.

        Args:
            node: The Node object to remove.
        """
        self.disconnect_node(node)
        if node in self.nodes:
            self.nodes.remove(node)

        # Update producer map
        for o in node.outputs:
            o_name = o.name if isinstance(o, Tensor) else o
            if self.producer_map.get(o_name) == node:
                del self.producer_map[o_name]

        # Update consumer map
        for i in node.inputs:
            i_name = i.name if isinstance(i, Tensor) else i
            if i_name in self.consumer_map and node in self.consumer_map[i_name]:
                self.consumer_map[i_name].remove(node)

    def __eq__(self, other: Any) -> bool:
        """Check equality with another Graph."""
        if not isinstance(other, Graph):
            return False
        if self.name != other.name:
            return False
        if len(self.nodes) != len(other.nodes):
            return False
        return all(n1 == n2 for n1, n2 in zip(self.nodes, other.nodes))

    def copy(self) -> "Graph":
        """Return a deep copy of the Graph.

        Returns:
            A new Graph object with all nodes and tensors copied.
        """
        g = Graph(self.name)
        g.doc_string = self.doc_string
        g.producer_name = self.producer_name
        g.producer_version = self.producer_version
        g.metadata_props = self.metadata_props.copy()
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
        g.value_info = [ValueInfo(v.name, v.shape, v.dtype) for v in self.value_info]
        g.initializers = self.initializers.copy()
        return g

    def tensors(self):
        """Return the dictionary of tensors in the graph."""
        return self.tensors

    def get_node(self, name: str) -> Optional[Node]:
        """Find a node by its name.

        Args:
            name: The name of the node to find.

        Returns:
            The Node object if found, else None.
        """
        for n in self.nodes:
            if n.name == name:
                return n
        return None

    def print_visualizer(self) -> None:
        """Print a visual summary of the graph topology to the log."""
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
