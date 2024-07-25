#include "CSINN2Reduction.hpp"

namespace MNN {

CSINN2Reduction::CSINN2Reduction(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : CSINN2CommonExecution(b, op) {
}

ErrorCode CSINN2Reduction::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
}

REGISTER_CSINN2_OP_CREATOR(CSINN2Reduction, OpType_Reduction)
} // namespace MNN
