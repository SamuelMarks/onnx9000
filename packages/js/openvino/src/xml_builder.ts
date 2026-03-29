export class XmlNode {
  name: string;
  attributes: Record<string, string>;
  children: (XmlNode | string)[];

  constructor(
    name: string,
    attributes: Record<string, string> = {},
    children: (XmlNode | string)[] = [],
  ) {
    this.name = name;
    this.attributes = attributes;
    this.children = children;
  }

  setAttribute(key: string, value: string): this {
    this.attributes[key] = value;
    return this;
  }

  addChild(child: XmlNode | string): this {
    this.children.push(child);
    return this;
  }

  toString(indent: number = 0, pretty: boolean = false): string {
    const indentStr = pretty ? ' '.repeat(indent) : '';
    const newline = pretty ? '\n' : '';

    let attrStr = '';
    for (const [key, value] of Object.entries(this.attributes)) {
      let escapedValue = String(value)
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

export class XmlBuilder {
  root: XmlNode | null = null;
  declaration: string = '<?xml version="1.0" ?>';

  constructor(rootName?: string) {
    if (rootName) {
      this.root = new XmlNode(rootName);
    }
  }

  setRoot(node: XmlNode): this {
    this.root = node;
    return this;
  }

  setDeclaration(dec: string): this {
    this.declaration = dec;
    return this;
  }

  toString(pretty: boolean = false): string {
    const newline = pretty ? '\n' : '';
    let result = this.declaration ? `${this.declaration}${newline}` : '';
    if (this.root) {
      result += this.root.toString(0, pretty);
    }
    return pretty ? result.trimEnd() : result;
  }
}
