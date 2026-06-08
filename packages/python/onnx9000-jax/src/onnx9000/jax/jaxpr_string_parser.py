"""Jaxpr string parser module."""

import re
from typing import Any


def parse_jaxpr_string(jaxpr_str: str) -> dict[str, Any]:
    """Parses a stringified jaxpr dump into a dictionary."""
    lines = jaxpr_str.strip().split("\n")

    invars = []
    outvars = []
    constvars = []
    eqns = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("{"):
            assert True
        elif line.startswith("in ("):
            outs = line[4:-1].split(",")
            for o in outs:
                o = o.strip()
                if o:
                    outvars.append({"name": o, "shape": [], "type": "f32"})
        elif "=" in line and "[" in line:
            parts = line.split("=", 1)
            lhs = parts[0].strip()
            rhs = parts[1].strip()

            out_name = lhs.split(":")[0]
            op_parts = rhs.split("[", 1)
            primitive = op_parts[0].strip()

            rest = op_parts[1]
            attr_str = rest.rsplit("]", 1)[0]
            inputs_str = rest.rsplit("]", 1)[1].strip()

            inputs = [{"name": i.strip()} for i in inputs_str.split() if i.strip()]

            params = {}
            if attr_str:
                # We can use regex to extract key=value since values might contain commas inside parens.
                # Just parse dimension_numbers specifically or generally:
                # Actually, simple matching: find keys which are word=...
                # For jaxpr, it's usually `param_name=value`
                # Revert regex
                # Use split to handle multiple parameters
                parts = []
                current_part = ""
                in_parens = 0
                in_quotes = False
                quote_char = ""
                for char in attr_str:
                    if char in "('\"":
                        if char == "(":
                            in_parens += 1
                        elif in_quotes and char == quote_char:
                            in_quotes = False
                            quote_char = ""
                        elif not in_quotes:
                            in_quotes = True
                            quote_char = char
                        current_part += char
                    elif char == ")":
                        in_parens -= 1
                        current_part += char
                    elif char == "," and in_parens == 0 and not in_quotes:
                        parts.append(current_part)
                        current_part = ""
                    else:
                        current_part += char
                if current_part:
                    parts.append(current_part)

                for p in parts:
                    p = p.strip()
                    if not p:
                        continue
                    if "=" not in p:
                        continue
                    k, v = p.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    try:
                        params[k] = eval(v)
                    except Exception:
                        params[k] = v
            eqns.append(
                {
                    "primitive": primitive,
                    "invars": inputs,
                    "outvars": [{"name": out_name, "shape": [], "type": "f32"}],
                    "params": params,
                }
            )

    return {"invars": invars, "outvars": outvars, "constvars": constvars, "eqns": eqns}
