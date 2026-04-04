"""Module docstring."""

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
            pass
        elif line.startswith("in ("):
            outs = line[4:-2].split(",")
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
                matches = re.finditer(r"([a-zA-Z_]+)=(.+?)(?=(?:, [a-zA-Z_]+=|$))", attr_str)
                for m in matches:
                    k = m.group(1).strip()
                    v = m.group(2).strip()
                    if v.endswith(","):
                        v = v[:-1]
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
