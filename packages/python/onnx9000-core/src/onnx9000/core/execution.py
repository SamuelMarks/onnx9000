"""Base abstractions for Execution Providers and Session Context."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from onnx9000.core.ir import Graph, Tensor


@dataclass
class SessionOptions:
    """Options for configuring an InferenceSession."""

    graph_optimization_level: int = 1
    optimized_model_filepath: str = ""
    enable_profiling: bool = False
    profile_file_prefix: str = "onnx9000_profile"
    execution_mode: str = "SEQUENTIAL"
    inter_op_num_threads: int = 0
    intra_op_num_threads: int = 0
    log_severity_level: int = 2
    logid: str = ""
    add_session_config_entry: dict[str, str] = field(default_factory=dict)
    register_custom_ops_library: list[str] = field(default_factory=list)
    free_dimension_overrides: dict[str, int] = field(default_factory=dict)


@dataclass
class RunOptions:
    """Options for configuring a specific run."""

    log_severity_level: int = 2
    log_verbosity_level: int = 0
    logid: str = ""
    run_tag: str = ""
    terminate: bool = False
    only_execute_path_to_fetches: bool = False


class Environment:
    """Singleton Environment equivalent for ORT."""

    _instance = None

    def __new__(cls) -> "Environment":
        """Singleton pattern for Environment."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def get_device() -> str:
        """Returns the native device name/type."""
        return "CPU"


@dataclass
class ExecutionContext:
    """Runtime context passed to Execution Providers during evaluation."""

    options: SessionOptions
    device_id: int = 0
    stream: Any = None


class ExecutionProvider(ABC):
    """Abstract base class for all hardware execution providers."""

    def __init__(self, options: dict[str, Any]) -> None:
        """Initialize the execution provider with specific options."""
        self.options = options
        self.device_id = int(options.get("device_id", 0))

    @abstractmethod
    def get_supported_nodes(self, graph: Graph) -> list[str]:
        """
        Return a list of node names from the graph that this provider can execute.
        """
        return []

    @abstractmethod
    def allocate_tensors(self, tensors: list[Tensor]) -> None:
        """
        Allocate necessary buffers for the provided tensors on the target device.
        """
        return None

    @abstractmethod
    def execute(
        self, graph: Graph, context: ExecutionContext, inputs: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """
        Execute the graph (or a supported subgraph) and return the outputs.
        """
        return {}
