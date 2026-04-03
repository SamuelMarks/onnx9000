"""Provides XML building utilities."""

from typing import Optional, Union


class XmlNode:
    """Represents an XML node."""

    def __init__(
        self,
        name: str,
        attributes: Optional[dict[str, str]] = None,
        children: Optional[list[Union["XmlNode", str]]] = None,
    ):
        """Initialize the XML node."""
        self.name = name
        self.attributes = attributes or {}
        self.children = children or []

    def set_attribute(self, key: str, value: str) -> "XmlNode":
        """Set an attribute on the node."""
        self.attributes[key] = str(value)
        return self

    def add_child(self, child: Union["XmlNode", str]) -> "XmlNode":
        """Add a child node or text."""
        self.children.append(child)
        return self

    def to_string(self, indent: int = 0, pretty: bool = False) -> str:
        """Convert the node to an XML string."""
        indent_str = " " * indent if pretty else ""
        newline = "\n" if pretty else ""

        attr_str = ""
        for key, value in self.attributes.items():
            escaped_value = (
                str(value)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&apos;")
            )
            attr_str += f' {key}="{escaped_value}"'

        if not self.children:
            return f"{indent_str}<{self.name}{attr_str} />{newline}"

        if len(self.children) == 1 and isinstance(self.children[0], str):
            escaped_text = (
                self.children[0].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )
            return f"{indent_str}<{self.name}{attr_str}>{escaped_text}</{self.name}>{newline}"

        result = f"{indent_str}<{self.name}{attr_str}>{newline}"
        child_indent = indent + 4 if pretty else 0

        for child in self.children:
            if isinstance(child, str):
                escaped_text = child.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                result += f"{' ' * child_indent if pretty else ''}{escaped_text}{newline}"
            else:
                result += child.to_string(child_indent, pretty)

        result += f"{indent_str}</{self.name}>{newline}"
        return result


class XmlBuilder:
    """Builds an XML document."""

    def set_root(self, node):
        """Set the root node."""
        self.root = node
        return self

    def set_declaration(self, decl):
        """Set the XML declaration."""
        self.declaration = decl
        return self

    def __init__(self, root_name: Optional[str] = None):
        """Initialize the builder."""
        self.root: Optional[XmlNode] = XmlNode(root_name) if root_name else None
        self.declaration = '<?xml version="1.0" ?>'

    def to_string(self, pretty: bool = False) -> str:
        """Convert the document to a string."""
        newline = "\n" if pretty else ""
        result = f"{self.declaration}{newline}" if self.declaration else ""
        if self.root:
            result += self.root.to_string(0, pretty)
        return result.rstrip() if pretty else result
