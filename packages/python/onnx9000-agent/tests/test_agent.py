"""Comprehensive unit tests for onnx9000.agent and onnx9000_agent."""

import onnx9000_agent
import pytest
from onnx9000.agent import Agent, AgentConfig, AgentMessage, AgentTool, run


def test_agent_run_and_tools() -> None:
    """Verify default run(), prompt execution, tool invocation, and error handling."""
    assert run() == "[agent] processed"
    assert onnx9000_agent.run() == "[agent] processed"
    assert run("hello") == "[onnx9000-agent] processed: hello"

    cfg = AgentConfig(name="custom-agent", system_prompt="System init")
    agent = Agent(cfg)
    assert len(agent.config.history) == 1
    assert agent.config.history[0] == AgentMessage(role="system", content="System init")

    agent.register_tool(
        AgentTool(
            name="echo",
            description="Echoes input text in uppercase",
            handler=lambda s: s.upper(),
        )
    )
    assert agent.list_tools() == ["echo"]

    tool_reply = agent.step("call:echo:onnx")
    assert tool_reply.content == "[custom-agent] echo -> ONNX"
    assert agent.config.history[-2].role == "tool"
    assert agent.config.history[-2].tool_name == "echo"

    with pytest.raises(KeyError, match="Unknown agent tool: missing"):
        agent.step("call:missing:arg")

    empty_agent = Agent(AgentConfig(system_prompt=""))
    assert empty_agent.config.history == []
