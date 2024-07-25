#include "CSINN2Binary.hpp"

namespace MNN {


CSINN2Binary::CSINN2Binary(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : CSINN2CommonExecution(b, op) {

}

ErrorCode CSINN2Binary::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

}


REGISTER_CSINN2_OP_CREATOR(CSINN2Binary, OpType_BinaryOp)
REGISTER_CSINN2_OP_CREATOR(CSINN2Binary, OpType_Eltwise)

} // namespace MNN
