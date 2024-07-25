#include "CSINN2Activation.hpp"

namespace MNN {


CSINN2Activation::CSINN2Activation(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : CoreMLCommonExecution(b, op) {
}

ErrorCode CSINN2Activation::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
}

REGISTER_CSINN2_OP_CREATOR(CSINN2Activation, OpType_ReLU)
REGISTER_CSINN2_OP_CREATOR(CSINN2Activation, OpType_ReLU6)
REGISTER_CSINN2_OP_CREATOR(CSINN2Activation, OpType_ELU)
REGISTER_CSINN2_OP_CREATOR(CSINN2Activation, OpType_PReLU)
REGISTER_CSINN2_OP_CREATOR(CSINN2Activation, OpType_Sigmoid)
REGISTER_CSINN2_OP_CREATOR(CSINN2Activation, OpType_Softmax)
} // namespace MNN
