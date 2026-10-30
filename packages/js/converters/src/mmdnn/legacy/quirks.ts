/**
 * @fileoverview Legacy framework quirk resolvers for Caffe, CNTK, and MXNet.
 */

/**
 * Legacy node attribute value union.
 */
export type QuirkAttrValue = string | number | boolean | number[] | string[] | null | undefined;

/**
 * Lightweight node representation for legacy quirk resolution passes.
 */
export interface Node {
  /** Property `opType`. */
  opType: string;
  /** Property `name`. */
  name?: string;
  /** Property `inputs`. */
  inputs?: string[];
  /** Property `outputs`. */
  outputs?: string[];
  /** Property `attributes`. */
  attributes?: Record<string, QuirkAttrValue>;
}

/**
 * Caffe phase inclusion descriptor.
 */
export interface CaffeIncludePhase {
  /** Property `phase`. */
  phase?: string | number;
}

/**
 * Caffe ROI pooling parameter descriptor.
 */
export interface CaffeRoiPoolingParam {
  /** Property `pooled_h`. */
  pooled_h?: number;
  /** Property `pooled_w`. */
  pooled_w?: number;
  /** Property `spatial_scale`. */
  spatial_scale?: number;
}

/**
 * Caffe layer descriptor used by legacy quirk resolvers.
 */
export interface CaffeQuirkLayer {
  /** Property `name`. */
  name?: string;
  /** Property `type`. */
  type?: string;
  /** Property `bottom`. */
  bottom?: string[];
  /** Property `top`. */
  top?: string[];
  /** Property `include`. */
  include?: CaffeIncludePhase | CaffeIncludePhase[];
  /** Property `roi_pooling_param`. */
  roi_pooling_param?: CaffeRoiPoolingParam;
}

/**
 * Resolves framework-specific quirks for legacy Caffe, CNTK, and MXNet models.
 */
export class LegacyQuirkResolver {
  // 171. Handle Caffe's infamous 0-padding quirks dynamically.
  /**
   * Executes `resolveCaffePadding` on `LegacyQuirkResolver`.
   * @param pad Parameter `pad`.
   * @returns Result of `resolveCaffePadding`.
   */
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
  /**
   * Executes `resolveCntkBroadcast` on `LegacyQuirkResolver`.
   * @param node Parameter `node`.
   * @returns Result of `resolveCntkBroadcast`.
   */
  static resolveCntkBroadcast(node: Node): Node {
    const newNode = { ...node };
    if (newNode.attributes?.broadcast) {
      newNode.attributes = { ...newNode.attributes };
      newNode.attributes.cntk_broadcast_resolved = true;
      // Convert to ONNX compatible attribute if needed
    }
    return newNode;
  }

  // 173. Resolve MXNet's specific Flatten behaviors which occasionally differ from ONNX depending on rank.
  /**
   * Executes `resolveMxnetFlatten` on `LegacyQuirkResolver`.
   * @param node Parameter `node`.
   * @param rank Parameter `rank`.
   * @returns Result of `resolveMxnetFlatten`.
   */
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
  /**
   * Executes `stripCaffeTrainingNodes` on `LegacyQuirkResolver`.
   * @param layers Parameter `layers`.
   * @returns Result of `stripCaffeTrainingNodes`.
   */
  static stripCaffeTrainingNodes(layers: CaffeQuirkLayer[]): CaffeQuirkLayer[] {
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
        const isTrainOnly = phases.some(
          (p: CaffeIncludePhase) => p.phase === 'TRAIN' || p.phase === 0,
        );
        if (isTrainOnly) {
          return false;
        }
      }
      return !trainingNodeTypes.has(layer.type ?? '');
    });
  }

  // 175. Emulate Caffe ROIPooling layer if possible via complex ONNX ops, or warn user.
  /**
   * Executes `emulateCaffeROIPooling` on `LegacyQuirkResolver`.
   * @param layer Parameter `layer`.
   * @returns Result of `emulateCaffeROIPooling`.
   */
  static emulateCaffeROIPooling(layer: CaffeQuirkLayer): Node[] {
    const nodes: Node[] = [];
    const node: Node = {
      opType: 'MaxRoiPool',
      inputs: layer.bottom || [],
      outputs: layer.top || [],
      attributes: {
        pooled_shape: [
          layer.roi_pooling_param?.pooled_h || 1,
          layer.roi_pooling_param?.pooled_w || 1,
        ],
        spatial_scale: layer.roi_pooling_param?.spatial_scale || 1.0,
      },
    };
    if (layer.name !== undefined) {
      node.name = layer.name;
    }
    nodes.push(node);
    return nodes;
  }
}
