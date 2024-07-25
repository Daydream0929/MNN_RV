#include "CSINN2LayerNorm.hpp"

namespace MNN {

CSINN2LayerNorm::CSINN2LayerNorm(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : CSINN2CommonExecution(b, op) {
}

ErrorCode CSINN2LayerNorm::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
}

REGISTER_CSINN2_OP_CREATOR(CSINN2LayerNorm, OpType_LayerNorm)
} // namespace MNN
