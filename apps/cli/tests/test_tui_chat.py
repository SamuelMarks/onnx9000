"""Unit tests and verification suite for tui chat functionality."""

from unittest.mock import patch

from tui_chat import start_chat_tui


@patch("builtins.input", side_effect=["hello", "exit"])
def test_tui_chat_exit(
    mock_input: int | float | str | bool | bytes | list | dict | tuple | None,
    capsys: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Verify that tui chat exit behaves as expected.

    Args:
        mock_input: Parameter specifying mock input.
        capsys: Parameter specifying capsys.
    """
    assert start_chat_tui() is True
    out, _ = capsys.readouterr()
    assert "Goodbye!" in out
    assert "mock" in out


@patch("builtins.input", side_effect=EOFError)
def test_tui_chat_eof(
    mock_input: int | float | str | bool | bytes | list | dict | tuple | None,
    capsys: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Verify that tui chat eof behaves as expected.

    Args:
        mock_input: Parameter specifying mock input.
        capsys: Parameter specifying capsys.
    """
    assert start_chat_tui() is True


@patch("builtins.input", side_effect=["", "quit"])
def test_tui_chat_empty_and_quit(
    mock_input: int | float | str | bool | bytes | list | dict | tuple | None,
    capsys: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Verify that tui chat empty and quit behaves as expected.

    Args:
        mock_input: Parameter specifying mock input.
        capsys: Parameter specifying capsys.
    """
    assert start_chat_tui() is True
    out, _ = capsys.readouterr()
    assert "Goodbye!" in out


@patch("builtins.input", side_effect=KeyboardInterrupt)
def test_tui_chat_keyboard_interrupt(
    mock_input: int | float | str | bool | bytes | list | dict | tuple | None,
    capsys: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Verify that tui chat keyboard interrupt behaves as expected.

    Args:
        mock_input: Parameter specifying mock input.
        capsys: Parameter specifying capsys.
    """
    assert start_chat_tui() is True
    out, _ = capsys.readouterr()
    assert "Goodbye!" in out
