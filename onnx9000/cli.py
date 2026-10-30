"""CLI entry point wrapper for the ONNX9000 command line interface.

This module delegates directly to the `onnx9000_cli` workspace package and falls
back to local workspace package path resolution if invoked outside an installed
editable environment.
"""

import os
import sys


def main() -> None:
    """Execute the ONNX9000 CLI main function."""
    try:
        from onnx9000_cli.main import main as cli_main

        cli_main()
    except ImportError:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cli_path = os.path.join(base_dir, "apps", "cli", "src")

        if os.path.exists(cli_path) and cli_path not in sys.path:
            sys.path.insert(0, cli_path)

        packages_dir = os.path.join(base_dir, "packages", "python")
        if os.path.exists(packages_dir):
            for pkg_name in os.listdir(packages_dir):
                pkg_src = os.path.join(packages_dir, pkg_name, "src")
                if os.path.exists(pkg_src) and pkg_src not in sys.path:
                    sys.path.insert(0, pkg_src)

        try:
            from onnx9000_cli.main import main as cli_main

            cli_main()
        except ImportError as exc:
            print(f"Error: ONNX9000 CLI is not properly installed ({exc}).")
            print(
                "To use the CLI in development, ensure you run `uv sync --all-packages` or install `apps/cli`."
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
