#include "CSINN2Unary.hpp"

namespace MNN {


CSINN2Unary::CSINN2Unary(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : CSINN2CommonExecution(b, op) {
}

ErrorCode CSINN2Unary::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
}

REGISTER_CSINN2_OP_CREATOR(CSINN2Unary, OpType_UnaryOp)
} // namespace MNN
