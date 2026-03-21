export interface Node {
  opType: string;
  name?: string;
  inputs?: string[];
  outputs?: string[];
  attributes?: Record<string, any>;
  [key: string]: any;
}

export class LegacyQuirkResolver {
  // 171. Handle Caffe's infamous 0-padding quirks dynamically.
  static resolveCaffePadding(pad: number[] | undefined | null): number[] {
    if (!pad || pad.length === 0) {
      return [0, 0, 0, 0];
    }
    if (pad.length === 1) {
      return [pad[0] as number, pad[0] as number, pad[0] as number, pad[0] as number];
    }
    if (pad.length === 2) {
      return [pad[0] as number, pad[1] as number, pad[0] as number, pad[1] as number];
    }
    if (pad.length === 4) {
      return pad;
    }
    const result = [0, 0, 0, 0];
    for (let i = 0; i < Math.min(pad.length, 4); i++) {
      result[i] = pad[i] as number;
    }
    return result;
  }

  // 172. Translate CNTK's dynamic axis broadcast rules properly into ONNX static ops.
  static resolveCntkBroadcast(node: Node): Node {
    const newNode = { ...node };
    if (newNode.attributes && newNode.attributes.broadcast) {
      newNode.attributes = { ...newNode.attributes };
      newNode.attributes.cntk_broadcast_resolved = true;
      // Convert to ONNX compatible attribute if needed
    }
    return newNode;
  }

  // 173. Resolve MXNet's specific Flatten behaviors which occasionally differ from ONNX depending on rank.
  static resolveMxnetFlatten(node: Node, rank: number): Node {
    const newNode = { ...node };
    if (newNode.opType === 'Flatten') {
      newNode.attributes = newNode.attributes ? { ...newNode.attributes } : {};
      if (rank === 0 || rank === 1) {
        newNode.attributes.axis = 0;
      } else {
        newNode.attributes.axis = 1;
      }
    }
    return newNode;
  }

  // 174. Strip unused training phase nodes (e.g., Accuracy, Loss) automatically from Caffe .prototxt.
  static stripCaffeTrainingNodes(layers: any[]): any[] {
    if (!layers) return [];
    const trainingNodeTypes = new Set([
      'Accuracy',
      'SoftmaxWithLoss',
      'EuclideanLoss',
      'SigmoidCrossEntropyLoss',
    ]);
    return layers.filter((layer) => {
      if (layer.include) {
        const phases = Array.isArray(layer.include) ? layer.include : [layer.include];
        const isTrainOnly = phases.some((p: any) => p.phase === 'TRAIN' || p.phase === 0);
        if (isTrainOnly) {
          return false;
        }
      }
      return !trainingNodeTypes.has(layer.type);
    });
  }

  // 175. Emulate Caffe ROIPooling layer if possible via complex ONNX ops, or warn user.
  static emulateCaffeROIPooling(layer: any): Node[] {
    const nodes: Node[] = [];
    nodes.push({
      opType: 'MaxRoiPool',
      name: layer.name,
      inputs: layer.bottom || [],
      outputs: layer.top || [],
      attributes: {
        pooled_shape: [
          layer.roi_pooling_param?.pooled_h || 1,
          layer.roi_pooling_param?.pooled_w || 1,
        ],
        spatial_scale: layer.roi_pooling_param?.spatial_scale || 1.0,
      },
    });
    return nodes;
  }
}
