import sys
import os


def main():
    try:
        from onnx9000_cli.main import main as cli_main

        cli_main()
    except ImportError:
        # Fallback for dev environment without installing apps/cli
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cli_path = os.path.join(base_dir, "apps", "cli", "src")

        # Inject cli source
        if os.path.exists(cli_path) and cli_path not in sys.path:
            sys.path.insert(0, cli_path)

        # Inject other python packages from workspace
        packages_dir = os.path.join(base_dir, "packages", "python")
        if os.path.exists(packages_dir):
            for p in os.listdir(packages_dir):
                pkg_src = os.path.join(packages_dir, p, "src")
                if os.path.exists(pkg_src) and pkg_src not in sys.path:
                    sys.path.insert(0, pkg_src)

        try:
            from onnx9000_cli.main import main as cli_main

            cli_main()
        except ImportError as e:
            print(f"Error: ONNX9000 CLI is not properly installed ({e}).")
            print(
                "To use the CLI in development, ensure you run uv pip install -e . or install pps/cli."
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
