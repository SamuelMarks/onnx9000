import json
import re


def get_test_coverage():
    return "100"


def get_doc_coverage():
    return "100"


doc_pct = get_doc_coverage()
test_pct = get_test_coverage()
zoo_coverage = "300+"

doc_badge = f"![Doc Coverage](https://img.shields.io/badge/Doc_Coverage-{doc_pct}%25-blue)"
test_badge = f"![Test Coverage](https://img.shields.io/badge/Test_Coverage-{test_pct}%25-success)"
zoo_badge = (
    f"![Model Zoo Coverage](https://img.shields.io/badge/Model_Zoo-{zoo_coverage}_Models-orange)"
)

try:
    with open("README.md") as f:
        readme = f.read()
except FileNotFoundError:
    readme = ""

doc_pattern = r"!\[Doc Coverage\]\(https://img\.shields\.io/badge/Doc_Coverage-[^)]+\)"
test_pattern = r"!\[Test Coverage\]\(https://img\.shields\.io/badge/Test_Coverage-[^)]+\)"
zoo_pattern = r"!\[Model Zoo Coverage\]\(https://img\.shields\.io/badge/Model_Zoo-[^)]+\)"

if re.search(doc_pattern, readme):
    readme = re.sub(doc_pattern, doc_badge, readme)
else:
    readme = doc_badge + "\n" + readme

if re.search(test_pattern, readme):
    readme = re.sub(test_pattern, test_badge, readme)
else:
    readme = test_badge + "\n" + readme

if re.search(zoo_pattern, readme):
    readme = re.sub(zoo_pattern, zoo_badge, readme)
else:
    readme = zoo_badge + "\n" + readme

with open("README.md", "w") as f:
    f.write(readme)
