/* eslint-disable */
/**
 * Represents an XML node for building OpenVINO IR files.
 */
export class XmlNode {
  /** The name of the XML tag. */
  name: string;
  /** Attributes of the XML tag. */
  attributes: Record<string, string>;
  /** Children of the XML tag, can be other XmlNodes or strings. */
  children: (XmlNode | string)[];

  /**
   * Creates a new XmlNode.
   * @param name - The name of the XML tag.
   * @param attributes - Initial attributes.
   * @param children - Initial children.
   */
  constructor(
    name: string,
    attributes: Record<string, string> = {},
    children: (XmlNode | string)[] = [],
  ) {
    this.name = name;
    this.attributes = attributes;
    this.children = children;
  }

  /**
   * Sets an attribute on the XML node.
   * @param key - The attribute name.
   * @param value - The attribute value.
   * @returns The current XmlNode for chaining.
   */
  setAttribute(key: string, value: string): this {
    this.attributes[key] = value;
    return this;
  }

  /**
   * Adds a child to the XML node.
   * @param child - The child to add.
   * @returns The current XmlNode for chaining.
   */
  addChild(child: XmlNode | string): this {
    this.children.push(child);
    return this;
  }

  /**
   * Converts the XML node to a string.
   * @param indent - The current indentation level.
   * @param pretty - Whether to format the output with newlines and indentation.
   * @returns The XML string representation.
   */
  toString(indent: number = 0, pretty: boolean = false): string {
    const indentStr = pretty ? ' '.repeat(indent) : '';
    const newline = pretty ? '\n' : '';

    let attrStr = '';
    for (const [key, value] of Object.entries(this.attributes)) {
      const escapedValue = String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&apos;');
      attrStr += ` ${key}="${escapedValue}"`;
    }

    if (this.children.length === 0) {
      return `${indentStr}<${this.name}${attrStr} />${newline}`;
    }

    if (this.children.length === 1 && typeof this.children[0] === 'string') {
      const escapedText = this.children[0]
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
      return `${indentStr}<${this.name}${attrStr}>${escapedText}</${this.name}>${newline}`;
    }

    let result = `${indentStr}<${this.name}${attrStr}>${newline}`;
    const childIndent = pretty ? indent + 4 : 0;

    for (const child of this.children) {
      if (typeof child === 'string') {
        const escapedText = child
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;');
        if (pretty) {
          result += ' '.repeat(childIndent) + escapedText + newline;
        } else {
          result += escapedText;
        }
      } else {
        result += child.toString(childIndent, pretty);
      }
    }

    result += `${indentStr}</${this.name}>${newline}`;
    return result;
  }
}

/**
 * A builder for creating XML documents.
 */
export class XmlBuilder {
  /** The root node of the XML document. */
  root: XmlNode | null = null;
  /** The XML declaration string. */
  declaration: string = '<?xml version="1.0" ?>';

  /**
   * Creates a new XmlBuilder.
   * @param rootName - Optional name for the root node.
   */
  constructor(rootName?: string) {
    if (rootName) {
      this.root = new XmlNode(rootName);
    }
  }

  /**
   * Sets the root node of the XML document.
   * @param node - The root node.
   * @returns The current XmlBuilder for chaining.
   */
  setRoot(node: XmlNode): this {
    this.root = node;
    return this;
  }

  /**
   * Sets the XML declaration string.
   * @param dec - The declaration string.
   * @returns The current XmlBuilder for chaining.
   */
  setDeclaration(dec: string): this {
    this.declaration = dec;
    return this;
  }

  /**
   * Converts the XML document to a string.
   * @param pretty - Whether to format the output with newlines and indentation.
   * @returns The XML document string representation.
   */
  toString(pretty: boolean = false): string {
    const newline = pretty ? '\n' : '';
    let result = this.declaration ? `${this.declaration}${newline}` : '';
    if (this.root) {
      result += this.root.toString(0, pretty);
    }
    return pretty ? result.trimEnd() : result;
  }
}
