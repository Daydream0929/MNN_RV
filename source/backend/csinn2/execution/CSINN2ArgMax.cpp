#include "CSINN2ArgMax.hpp"

namespace MNN {

CSINN2ArgMax::CSINN2ArgMax(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : CSINN2CommonExecution(b, op) {

}

ErrorCode CSINN2ArgMax::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

}

REGISTER_CSINN2_OP_CREATOR(CSINN2ArgMax, OpType_ArgMax)
} // namespace MNN
