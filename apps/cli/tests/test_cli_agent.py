import argparse
from unittest.mock import patch

from onnx9000_cli.main import agent_cmd


def test_agent_cmd():
    args = argparse.Namespace(task=["prune", "the", "graph"])
    with patch("builtins.print") as mock_print:
        agent_cmd(args)

        mock_print.assert_any_call('Starting agent workflow with task: "prune the graph"...')
        mock_print.assert_any_call("Reasoning...")
        mock_print.assert_any_call("Action: analyze_graph")
        mock_print.assert_any_call("Action: optimize_graph")
        mock_print.assert_any_call("Final Answer: Task completed successfully.")


def test_agent_cmd_empty_task():
    args = argparse.Namespace(task=[])
    with patch("builtins.print") as mock_print:
        agent_cmd(args)

        mock_print.assert_any_call('Starting agent workflow with task: ""...')
        mock_print.assert_any_call("Final Answer: Task completed successfully.")
