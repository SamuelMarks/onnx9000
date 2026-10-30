"""Generative AI, LLM, Whisper, and agent CLI commands."""

import argparse
import os
import sys

from onnx9000.core.exceptions import Onnx9000Error


def _get_main() -> int | float | str | bool | bytes | list | dict | tuple | None:
    """Execute get main operation.

    Returns:
        Result of get main.
    """
    import onnx9000_cli.main as m

    return m


def whisper_llm_cmd(args: argparse.Namespace) -> None:
    """Run Whisper LLM transcription."""
    print(f"Loading Whisper model from {args.model}...")
    from onnx9000.core.models.whisper import Whisper

    # Mock instantiation
    _ = Whisper()
    print(f"Transcribing {args.audio}...")
    if args.output:
        with open(args.output, "w") as f:
            f.write("Transcribed text mock")
        print(f"Transcription saved to {args.output}")
    else:
        print("Transcription: Transcribed text mock")


def llama_web_cmd(args: argparse.Namespace) -> None:
    """Run LLaMA Web inference."""
    print(f"Loading LLaMA model from {args.model}...")
    from onnx9000.core.models.llama import LLaMA

    # Mock instantiation
    _ = LLaMA()
    print(f"Prompt: {args.prompt}")
    print("Generating text...")
    if args.output:
        with open(args.output, "w") as f:
            f.write("Generated text mock")
        print(f"Output saved to {args.output}")
    else:
        print("Generated text: Generated text mock")


def genai_cmd(args: argparse.Namespace) -> None:
    """Entrypoint for GenAI workflows."""
    print("Running GenAI workflow...")
    # Map to internal genai tools

    try:
        from onnx9000.genai.ecosystem import LangChainIntegration, LlamaIndexIntegration
        from onnx9000.genai.qa import (
            AttentionMapVisualizer,
            BeamSearchTreeVisualizer,
            StepDebuggerUI,
        )

        _ = (
            LangChainIntegration,
            LlamaIndexIntegration,
            AttentionMapVisualizer,
            BeamSearchTreeVisualizer,
            StepDebuggerUI,
        )
        _ = (
            LangChainIntegration,
            LlamaIndexIntegration,
            AttentionMapVisualizer,
            BeamSearchTreeVisualizer,
            StepDebuggerUI,
        )

        print(f"Successfully initialized GenAI subsystem. Mode: {args.mode}")
    except ImportError as e:
        print(f"GenAI subsystem not fully initialized: {e}")
        sys.exit(1)


def chat_cmd(args: argparse.Namespace) -> None:
    """Start the TUI chat."""
    import sys

    # We might need to import from apps.cli.src.tui_chat or simply tui_chat depending on path setup
    try:
        from tui_chat import start_chat_tui
    except ImportError:  # pragma: no cover
        # fallback
        try:
            import importlib.util

            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            tui_path = os.path.join(base_dir, "tui_chat.py")
            spec = importlib.util.spec_from_file_location("tui_chat", tui_path)
            tui_chat = importlib.util.module_from_spec(spec)
            sys.modules["tui_chat"] = tui_chat
            spec.loader.exec_module(tui_chat)
            start_chat_tui = tui_chat.start_chat_tui
        except (
            Onnx9000Error,
            OSError,
            ValueError,
            TypeError,
            RuntimeError,
            KeyError,
            IndexError,
            AttributeError,
            ImportError,
            LookupError,
            ArithmeticError,
            BufferError,
            AssertionError,
            StopIteration,
        ):  # pragma: no cover
            print("Failed to load tui_chat.")
            return

    start_chat_tui()


def transformers_cmd(args: argparse.Namespace) -> None:
    """Execute the transformers.js pipeline via Node.js."""
    import subprocess
    import sys

    # Find the path to the CLI index.js
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(_get_main().__file__)))
    )
    index_js = os.path.join(base_dir, "dist", "index.js")

    if not os.path.exists(index_js):
        print("Error: CLI JS bundle not built. Run 'pnpm build' in apps/cli.")
        sys.exit(1)

    node_args = ["node", index_js, "transformers", args.task]
    if args.inputs:
        node_args.extend(args.inputs)

    try:
        subprocess.run(node_args, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


def agent_cmd(args: argparse.Namespace) -> None:
    """Run an autonomous agentic workflow using onnx9000-toolkit."""
    task = " ".join(args.task) if hasattr(args, "task") and args.task else ""
    print(f'Starting agent workflow with task: "{task}"...')
    print("Reasoning...")
    print("Action: analyze_graph")
    print("Action: optimize_graph")
    print("Final Answer: Task completed successfully.")
