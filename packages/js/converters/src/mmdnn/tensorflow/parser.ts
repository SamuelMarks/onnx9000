/**
 * Represents the value of an attribute attached to a TensorFlow node.
 */
export interface TFAttrValue {
  type?: string;
  shape?: number[];
  i?: number;
  f?: number;
  s?: string;
  b?: boolean;
  list?: { i?: number[]; f?: number[]; s?: string[] };
  tensor?: { dtype: string; shape: number[] };
}

/**
 * Represents a single parsed TensorFlow node from a .pbtxt file.
 */
export interface TFNodeDef {
  name: string;
  op: string;
  input: string[];
  attr: Record<string, TFAttrValue>;
}

/**
 * Represents the entire parsed TensorFlow computation graph.
 */
export interface TFGraphDef {
  node: TFNodeDef[];
}

/**
 * Parses a TensorFlow Text Protobuf (.pbtxt) string into a structured graph object.
 * This provides a lightweight alternative to loading the full protobuf definition
 * inside the browser.
 *
 * @param text The raw .pbtxt string.
 * @returns A structured TFGraphDef object.
 */
export function parsePbtxt(text: string): TFGraphDef {
  const result: TFGraphDef = { node: [] };
  const lines = text.split('\n');

  let currentNode: TFNodeDef | null = null;
  let currentAttrKey: string | null = null;
  let currentAttrVal: TFAttrValue | null = null;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!.trim();
    if (!line) continue;

    if (line.startsWith('node {')) {
      currentNode = { name: '', op: '', input: [], attr: {} };
      result.node.push(currentNode);
      continue;
    }

    if (!currentNode) continue;

    if (line.startsWith('name: "')) {
      currentNode.name = line.match(/name: "(.*)"/)?.[1] || '';
    } else if (line.startsWith('op: "')) {
      currentNode.op = line.match(/op: "(.*)"/)?.[1] || '';
    } else if (line.startsWith('input: "')) {
      currentNode.input.push(line.match(/input: "(.*)"/)?.[1] || '');
    } else if (line.startsWith('attr {')) {
      currentAttrVal = {};
      currentAttrKey = null;
    } else if (currentAttrVal && line.startsWith('key: "')) {
      currentAttrKey = line.match(/key: "(.*)"/)?.[1] || '';
    } else if (currentAttrVal && line === '}') {
      if (currentAttrKey) {
        currentNode.attr[currentAttrKey] = currentAttrVal;
        currentAttrKey = null;
        currentAttrVal = null;
      }
    } else if (currentAttrVal) {
      if (line.includes('type: DT_')) {
        currentAttrVal.type = line.match(/type: (DT_[A-Z0-9]+)/)?.[1] || 'DT_FLOAT';
      }
      if (line.includes('shape {') && !line.includes('tensor_shape {')) {
        currentAttrVal.shape = [];
        const dims = line.matchAll(/size: (-?\d+)/g);
        for (const d of dims) {
          currentAttrVal.shape.push(parseInt(d[1]!));
        }
      }
      if (line.includes('tensor_shape {')) {
        if (!currentAttrVal.tensor) currentAttrVal.tensor = { dtype: 'DT_FLOAT', shape: [] };
        const dims = line.matchAll(/size: (-?\d+)/g);
        for (const d of dims) {
          currentAttrVal.tensor.shape.push(parseInt(d[1]!));
        }
      }
      if (line.includes('s: "')) {
        currentAttrVal.s = line.match(/s: "(.*)"/)?.[1] || '';
      }
      if (line.includes('list {')) {
        if (line.includes('i: ')) {
          currentAttrVal.list = { i: [] };
          const ints = line.matchAll(/i: (-?\d+)/g);
          for (const val of ints) {
            currentAttrVal.list.i!.push(parseInt(val[1]!));
          }
        }
      }
      if (line.includes('i: ') && !line.includes('list {')) {
        const match = line.match(/i: (-?\d+)/);
        if (match) currentAttrVal.i = parseInt(match[1]!);
      }
      if (line.includes('f: ') && !line.includes('list {')) {
        const match = line.match(/f: (-?[\d.]+)/);
        if (match) currentAttrVal.f = parseFloat(match[1]!);
      }
    }
  }

  return result;
}
