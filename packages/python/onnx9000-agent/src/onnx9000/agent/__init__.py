"""Typed tool-calling agent orchestration runtime for ONNX9000."""

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentMessage:
    """Represents a structured conversational message in an agent session.

    Attributes:
        role: Sender role (`"system"`, `"user"`, `"assistant"`, or `"tool"`).
        content: Textual payload of the message.
        tool_name: Optional tool identifier when `role == "tool"`.
    """

    role: str
    content: str
    tool_name: str | None = None


@dataclass(frozen=True)
class AgentTool:
    """Registered callable tool exposed to an ONNX9000 `Agent`.

    Attributes:
        name: Unique tool identifier.
        description: Human-readable description of the tool's purpose.
        handler: Callable taking a string argument and returning a string result.
    """

    name: str
    description: str
    handler: Callable[[str], str]


@dataclass
class AgentConfig:
    """Configuration parameters for an `Agent` orchestration instance.

    Attributes:
        name: Agent identifier.
        system_prompt: Initial system prompt prepended to conversations.
        max_steps: Maximum number of tool-invocation iterations per turn.
    """

    name: str = "onnx9000-agent"
    system_prompt: str = "You are a helpful ONNX9000 assistant."
    max_steps: int = 4
    history: list[AgentMessage] = field(default_factory=list)


class Agent:
    """Tool-calling agent runtime orchestrating prompts, tools, and GenAI state."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize the agent with optional configuration.

        Args:
            config: Agent configuration; defaults to a standard `AgentConfig`.
        """
        self.config: AgentConfig = config if config is not None else AgentConfig()
        self._tools: dict[str, AgentTool] = {}
        if not self.config.history and self.config.system_prompt:
            self.config.history.append(
                AgentMessage(role="system", content=self.config.system_prompt)
            )

    def register_tool(self, tool: AgentTool) -> None:
        """Register a callable tool with the agent.

        Args:
            tool: The `AgentTool` definition to register.
        """
        self._tools[tool.name] = tool

    def list_tools(self) -> list[str]:
        """Return the names of all registered tools.

        Returns:
            List of registered tool names.
        """
        return list(self._tools.keys())

    def step(self, user_input: str) -> AgentMessage:
        """Execute a single agent reasoning and optional tool-calling step.

        If `user_input` starts with `"call:<tool_name>:<arg>"`, invokes the named
        tool and records both the tool output and assistant synthesis in history.

        Args:
            user_input: Prompt or tool directive from the user.

        Returns:
            The resulting assistant `AgentMessage`.

        Raises:
            KeyError: If a requested tool name is not registered.
        """
        self.config.history.append(AgentMessage(role="user", content=user_input))
        if user_input.startswith("call:"):
            parts = user_input.split(":", 2)
            tool_name = parts[1] if len(parts) > 1 else ""
            arg = parts[2] if len(parts) > 2 else ""
            if tool_name not in self._tools:
                raise KeyError(f"Unknown agent tool: {tool_name}")
            tool_result = self._tools[tool_name].handler(arg)
            self.config.history.append(
                AgentMessage(role="tool", content=tool_result, tool_name=tool_name)
            )
            reply = AgentMessage(
                role="assistant", content=f"[{self.config.name}] {tool_name} -> {tool_result}"
            )
        else:
            reply = AgentMessage(
                role="assistant", content=f"[{self.config.name}] processed: {user_input}"
            )
        self.config.history.append(reply)
        return reply


def run(prompt: str | None = None) -> str:
    """Execute the default agent runtime.

    Args:
        prompt: Optional user prompt to process. When `None`, returns default status.

    Returns:
        Formatted agent response string.
    """
    if prompt is None:
        return "[agent] processed"
    agent = Agent()
    return agent.step(prompt).content


__all__ = ["Agent", "AgentConfig", "AgentMessage", "AgentTool", "run"]
