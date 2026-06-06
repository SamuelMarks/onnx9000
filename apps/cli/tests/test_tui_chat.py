import pytest
from unittest.mock import patch
import sys
import os

# Ensure tui_chat is importable 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
import tui_chat

def test_start_chat_tui_exit():
    with patch("builtins.input", side_effect=["hello", "exit"]):
        with patch("builtins.print") as mock_print:
            assert tui_chat.start_chat_tui() == True
            mock_print.assert_any_call("ONNX9000 Assistant: I received 'hello', but I am a simple mock.")
            mock_print.assert_any_call("Goodbye!")

def test_start_chat_tui_eof():
    with patch("builtins.input", side_effect=EOFError):
        assert tui_chat.start_chat_tui() == True

def test_start_chat_tui_keyboard_interrupt():
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        assert tui_chat.start_chat_tui() == True

def test_start_chat_tui_empty_input():
    with patch("builtins.input", side_effect=["", "quit"]):
        assert tui_chat.start_chat_tui() == True
