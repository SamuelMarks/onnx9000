import argparse
from unittest.mock import MagicMock, mock_open, patch

from onnx9000_cli.main import llama_web_cmd, whisper_llm_cmd


def test_whisper_llm_cmd_stdout():
    args = argparse.Namespace(model="whisper.onnx", audio="test.wav", output=None)
    with (
        patch("onnx9000.core.models.whisper.Whisper") as mock_whisper,
        patch("builtins.print") as mock_print,
    ):
        whisper_llm_cmd(args)
    mock_whisper.assert_called_once()
    mock_print.assert_any_call("Transcription: Transcribed text mock")


def test_whisper_llm_cmd_file():
    args = argparse.Namespace(model="whisper.onnx", audio="test.wav", output="out.txt")
    m_open = mock_open()
    with (
        patch("onnx9000.core.models.whisper.Whisper") as mock_whisper,
        patch("builtins.open", m_open),
    ):
        whisper_llm_cmd(args)
    mock_whisper.assert_called_once()
    m_open.assert_called_once_with("out.txt", "w")
    m_open().write.assert_called_once_with("Transcribed text mock")


def test_llama_web_cmd_stdout():
    args = argparse.Namespace(model="llama.onnx", prompt="Hello", output=None)
    with (
        patch("onnx9000.core.models.llama.LLaMA") as mock_llama,
        patch("builtins.print") as mock_print,
    ):
        llama_web_cmd(args)
    mock_llama.assert_called_once()
    mock_print.assert_any_call("Generated text: Generated text mock")


def test_llama_web_cmd_file():
    args = argparse.Namespace(model="llama.onnx", prompt="Hello", output="out.txt")
    m_open = mock_open()
    with patch("onnx9000.core.models.llama.LLaMA") as mock_llama, patch("builtins.open", m_open):
        llama_web_cmd(args)
    mock_llama.assert_called_once()
    m_open.assert_called_once_with("out.txt", "w")
    m_open().write.assert_called_once_with("Generated text mock")
