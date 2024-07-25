#include "CSINN2Quant.hpp"

namespace MNN {


CSINN2Quant::CSINN2Quant(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : CSINN2CommonExecution(b, op) {
}

ErrorCode CSINN2Quant::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
}

CSINN2Dequant::CSINN2Dequant(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : CSINN2CommonExecution(b, op) {
}

ErrorCode CSINN2Dequant::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
}

REGISTER_CSINN2_OP_CREATOR(CSINN2Quant, OpType_FloatToInt8)
REGISTER_CSINN2_OP_CREATOR(CSINN2Dequant, OpType_Int8ToFloat)
} // namespace MNN