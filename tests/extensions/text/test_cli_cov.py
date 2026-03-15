"""Module providing core logic and structural definitions."""


def test_text_cli_exception(monkeypatch):
    """Provides semantic functionality and verification."""
    import onnx9000.extensions.text.cli
    import sys
    from unittest.mock import patch

    monkeypatch.setattr(sys, "argv", ["text-cli", "in.json", "out.bin"])
    with patch(
        "onnx9000.extensions.text.cli.export_tokenizer_binary",
        side_effect=ValueError("foo"),
    ):
        with patch("sys.exit") as mock_exit:
            onnx9000.extensions.text.cli.main()
            mock_exit.assert_called_with(1)
