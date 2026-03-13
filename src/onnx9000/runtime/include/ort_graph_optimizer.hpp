#pragma once

#include <string>
#include <vector>

#include "ort_env.hpp"

namespace onnx9000 {

struct GraphOptimizationState {
  bool enabled{true};
  std::string name;
};

class ORT_ENV_EXPORT GraphOptimizer {
public:
  virtual ~GraphOptimizer() = default;
  virtual const char *PassName() const noexcept = 0;
  virtual bool Apply() noexcept = 0; // Dummy apply
};

#define DECLARE_DUMMY_OPTIMIZER(Name)                                          \
  class ORT_ENV_EXPORT Name : public GraphOptimizer {                          \
  public:                                                                      \
    const char *PassName() const noexcept override { return #Name; }           \
    bool Apply() noexcept override { return true; }                            \
  };

DECLARE_DUMMY_OPTIMIZER(ConstantFolding)
DECLARE_DUMMY_OPTIMIZER(RedundantNodeElimination)
DECLARE_DUMMY_OPTIMIZER(CastElimination)
DECLARE_DUMMY_OPTIMIZER(IdentityElimination)
DECLARE_DUMMY_OPTIMIZER(UnsqueezeElimination)
DECLARE_DUMMY_OPTIMIZER(SliceElimination)
DECLARE_DUMMY_OPTIMIZER(DropoutElimination)
DECLARE_DUMMY_OPTIMIZER(NodeBreakFusion)
DECLARE_DUMMY_OPTIMIZER(ShapeToInitializer)
DECLARE_DUMMY_OPTIMIZER(ReshapeFusion)
DECLARE_DUMMY_OPTIMIZER(FreeDimensionOverride)

// Level 2 Fusions
DECLARE_DUMMY_OPTIMIZER(ConvActivationFusion)
DECLARE_DUMMY_OPTIMIZER(ConvBatchNormFusion)
DECLARE_DUMMY_OPTIMIZER(ConvAddFusion)
DECLARE_DUMMY_OPTIMIZER(MatMulAddFusion)
DECLARE_DUMMY_OPTIMIZER(MatMulScaleFusion)
DECLARE_DUMMY_OPTIMIZER(GemmActivationFusion)
DECLARE_DUMMY_OPTIMIZER(LayerNormFusion)
DECLARE_DUMMY_OPTIMIZER(SimplifiedLayerNormFusion)
DECLARE_DUMMY_OPTIMIZER(AttentionFusion)
DECLARE_DUMMY_OPTIMIZER(EmbedLayerNormFusion)
DECLARE_DUMMY_OPTIMIZER(BiasGeluFusion)
DECLARE_DUMMY_OPTIMIZER(FastGeluFusion)
DECLARE_DUMMY_OPTIMIZER(SkipLayerNormFusion)
DECLARE_DUMMY_OPTIMIZER(QLinearConvFusion)
DECLARE_DUMMY_OPTIMIZER(QLinearMatMulFusion)
DECLARE_DUMMY_OPTIMIZER(RotaryEmbeddingFusion)
DECLARE_DUMMY_OPTIMIZER(MultiHeadAttentionFusion)

// Level 3
DECLARE_DUMMY_OPTIMIZER(NCHW_to_NHWC_Transformation)
DECLARE_DUMMY_OPTIMIZER(NHWC_to_NCHW_Transformation)
DECLARE_DUMMY_OPTIMIZER(NCDHW_to_NDHWC_Transformation)

} // namespace onnx9000
