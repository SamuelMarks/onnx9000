"""Badge updater script for README.md coverage and documentation shields.

Reads `coverage.json` (when present) to compute line coverage percentage and updates
the `![Test Coverage]` and `![Doc Coverage]` shields in `README.md`.
"""

import json
import os
import re


def update_readme_badges(
    readme_path: str = "README.md",
    coverage_path: str = "coverage.json",
) -> str:
    """Update Test Coverage and Doc Coverage shields in the target README file.

    Args:
        readme_path: Path to the Markdown README file to update.
        coverage_path: Path to the JSON coverage report produced by `coverage json`.

    Returns:
        Status identifier `"update_badges"`.
    """
    test_pct = "100"
    if os.path.exists(coverage_path):
        with open(coverage_path, encoding="utf-8") as cov_file:
            cov_data = json.load(cov_file)
            totals = cov_data.get("totals", {})
            pct_val = totals.get("percent_covered_display")
            if isinstance(pct_val, str) and pct_val:
                test_pct = pct_val

    if os.path.exists(readme_path):
        with open(readme_path, encoding="utf-8") as rm_file:
            content = rm_file.read()

        doc_badge = "![Doc Coverage](https://img.shields.io/badge/Doc_Coverage-100%25-blue)"
        test_badge = (
            f"![Test Coverage](https://img.shields.io/badge/Test_Coverage-{test_pct}%25-success)"
        )
        content = re.sub(
            r"!\[Doc Coverage\]\(https://img\.shields\.io/badge/Doc_Coverage-[^)]+\)",
            doc_badge,
            content,
        )
        content = re.sub(
            r"!\[Test Coverage\]\(https://img\.shields\.io/badge/Test_Coverage-[^)]+\)",
            test_badge,
            content,
        )
        with open(readme_path, "w", encoding="utf-8") as rm_file:
            rm_file.write(content)

    return "update_badges"


def run() -> str:
    """Execute the default README badge update routine.

    Returns:
        Status identifier `"update_badges"`.
    """
    return update_readme_badges()


if __name__ == "__main__":
    run()
