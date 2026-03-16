"""Module containing fix_logger.py definitions."""

import glob

for f_path in glob.glob("src/**/*.py", recursive=True):
    with open(f_path) as f:
        content = f.read()
    content = content.replace(
        "import logging\\nlogger = logging.getLogger(__name__)",
        "import logging\nlogger = logging.getLogger(__name__)",
    )
    with open(f_path, "w") as f:
        f.write(content)
for f_path in glob.glob("tests/**/*.py", recursive=True):
    with open(f_path) as f:
        content = f.read()
    content = content.replace(
        "import logging\\nlogger = logging.getLogger(__name__)",
        "import logging\nlogger = logging.getLogger(__name__)",
    )
    with open(f_path, "w") as f:
        f.write(content)
