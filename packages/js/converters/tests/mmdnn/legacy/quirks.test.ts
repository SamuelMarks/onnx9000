import { describe, expect, it } from 'vitest';
import { LegacyQuirkResolver } from '../../../src/mmdnn/legacy/quirks';

describe('LegacyQuirkResolver', () => {
  describe('resolveCaffePadding', () => {
    it('should return [0, 0, 0, 0] for empty or null pad', () => {
      expect(LegacyQuirkResolver.resolveCaffePadding([])).toEqual([0, 0, 0, 0]);
      expect(LegacyQuirkResolver.resolveCaffePadding(null)).toEqual([0, 0, 0, 0]);
      expect(LegacyQuirkResolver.resolveCaffePadding(undefined)).toEqual([0, 0, 0, 0]);
    });

    it('should expand single value padding symmetrically', () => {
      expect(LegacyQuirkResolver.resolveCaffePadding([1])).toEqual([1, 1, 1, 1]);
    });

    it('should expand two value padding symmetrically', () => {
      expect(LegacyQuirkResolver.resolveCaffePadding([1, 2])).toEqual([1, 2, 1, 2]);
    });

    it('should return 4-element padding as-is', () => {
      expect(LegacyQuirkResolver.resolveCaffePadding([1, 2, 3, 4])).toEqual([1, 2, 3, 4]);
    });

    it('should pad with 0s if length is unusual', () => {
      expect(LegacyQuirkResolver.resolveCaffePadding([1, 2, 3])).toEqual([1, 2, 3, 0]);
    });
  });

  describe('resolveCntkBroadcast', () => {
    it('should add cntk_broadcast_resolved true if broadcast exists', () => {
      const node = { opType: 'Add', attributes: { broadcast: 1 } };
      const resolved = LegacyQuirkResolver.resolveCntkBroadcast(node);
      expect(resolved.attributes?.cntk_broadcast_resolved).toBe(true);
    });

    it('should not alter node if broadcast is not present', () => {
      const node = { opType: 'Add', attributes: { other_attr: 1 } };
      const resolved = LegacyQuirkResolver.resolveCntkBroadcast(node);
      expect(resolved.attributes?.cntk_broadcast_resolved).toBeUndefined();
    });
  });

  describe('resolveMxnetFlatten', () => {
    it('should set axis to 0 for rank 0 or 1', () => {
      const node = { opType: 'Flatten' };
      const resolvedRank0 = LegacyQuirkResolver.resolveMxnetFlatten(node, 0);
      expect(resolvedRank0.attributes?.axis).toBe(0);

      const resolvedRank1 = LegacyQuirkResolver.resolveMxnetFlatten(node, 1);
      expect(resolvedRank1.attributes?.axis).toBe(0);
    });

    it('should set axis to 1 for rank > 1', () => {
      const node = { opType: 'Flatten' };
      const resolvedRank2 = LegacyQuirkResolver.resolveMxnetFlatten(node, 2);
      expect(resolvedRank2.attributes?.axis).toBe(1);
    });

    it('should not modify non-Flatten nodes', () => {
      const node = { opType: 'Add' };
      const resolved = LegacyQuirkResolver.resolveMxnetFlatten(node, 2);
      expect(resolved.attributes?.axis).toBeUndefined();
    });
  });

  describe('stripCaffeTrainingNodes', () => {
    it('should remove explicitly listed training nodes', () => {
      const layers = [
        { type: 'Convolution' },
        { type: 'Accuracy' },
        { type: 'SoftmaxWithLoss' },
        { type: 'EuclideanLoss' },
        { type: 'SigmoidCrossEntropyLoss' },
        { type: 'Pooling' },
      ];
      const stripped = LegacyQuirkResolver.stripCaffeTrainingNodes(layers);
      expect(stripped.length).toBe(2);
      expect(stripped[0].type).toBe('Convolution');
      expect(stripped[1].type).toBe('Pooling');
    });

    it('should remove nodes explicitly marked for TRAIN phase', () => {
      const layers = [
        { type: 'Convolution' },
        { type: 'Dropout', include: { phase: 'TRAIN' } },
        { type: 'Dropout', include: [{ phase: 0 }] }, // 0 is TRAIN in Caffe enum
      ];
      const stripped = LegacyQuirkResolver.stripCaffeTrainingNodes(layers);
      expect(stripped.length).toBe(1);
      expect(stripped[0].type).toBe('Convolution');
    });

    it('should keep nodes for TEST phase', () => {
      const layers = [{ type: 'Convolution' }, { type: 'Dropout', include: { phase: 'TEST' } }];
      const stripped = LegacyQuirkResolver.stripCaffeTrainingNodes(layers);
      expect(stripped.length).toBe(2);
    });
  });

  describe('emulateCaffeROIPooling', () => {
    it('should convert Caffe ROIPooling to ONNX MaxRoiPool', () => {
      const layer = {
        name: 'roi_pool',
        type: 'ROIPooling',
        bottom: ['features', 'rois'],
        top: ['pool'],
        roi_pooling_param: {
          pooled_h: 7,
          pooled_w: 7,
          spatial_scale: 0.0625,
        },
      };

      const nodes = LegacyQuirkResolver.emulateCaffeROIPooling(layer);
      expect(nodes.length).toBe(1);

      const onnxNode = nodes[0];
      expect(onnxNode.opType).toBe('MaxRoiPool');
      expect(onnxNode.name).toBe('roi_pool');
      expect(onnxNode.inputs).toEqual(['features', 'rois']);
      expect(onnxNode.outputs).toEqual(['pool']);
      expect(onnxNode.attributes?.pooled_shape).toEqual([7, 7]);
      expect(onnxNode.attributes?.spatial_scale).toBe(0.0625);
    });

    it('should use default values if roi_pooling_param is missing', () => {
      const layer = {
        name: 'roi_pool',
        type: 'ROIPooling',
      };

      const nodes = LegacyQuirkResolver.emulateCaffeROIPooling(layer);
      expect(nodes[0].attributes?.pooled_shape).toEqual([1, 1]);
      expect(nodes[0].attributes?.spatial_scale).toBe(1.0);
    });
  });
});
