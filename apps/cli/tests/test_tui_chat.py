from unittest.mock import patch

from tui_chat import start_chat_tui


@patch("builtins.input", side_effect=["hello", "exit"])
def test_tui_chat_exit(mock_input, capsys):
    assert start_chat_tui() is True
    out, _ = capsys.readouterr()
    assert "Goodbye!" in out
    assert "mock" in out


@patch("builtins.input", side_effect=EOFError)
def test_tui_chat_eof(mock_input, capsys):
    assert start_chat_tui() is True


@patch("builtins.input", side_effect=["", "quit"])
def test_tui_chat_empty_and_quit(mock_input, capsys):
    assert start_chat_tui() is True
    out, _ = capsys.readouterr()
    assert "Goodbye!" in out


@patch("builtins.input", side_effect=KeyboardInterrupt)
def test_tui_chat_keyboard_interrupt(mock_input, capsys):
    assert start_chat_tui() is True
    out, _ = capsys.readouterr()
    assert "Goodbye!" in out
