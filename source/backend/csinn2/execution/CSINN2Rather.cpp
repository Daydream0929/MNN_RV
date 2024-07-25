#include "CSINN2Gather.hpp"

namespace MNN {


CSINN2Gather::CSINN2Gather(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : CSINN2CommonExecution(b, op) {
}

ErrorCode CSINN2Gather::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    int axis = 0;
    if (inputs.size() == 3) {
        auto axis_tensor = inputs[2];
        axis = axis_tensor->host<int32_t>()[0];
    }
    if (mOp->main_type() == OpParameter_Axis) {
        axis = mOp->main_as_Axis()->axis();
    }
    if (axis < 0) {
        axis = input->buffer().dimensions + axis;
    }
    // gather: [input, axis, indices]

    /* CSINN2 
    auto inputIdx   = mCSINN2Backend->getTensorIdx(inputs[0]);
    auto axisIdx    = buildScalar(formatAxis(axis, input));
    auto indicesIdx = mCSINN2Backend->getTensorIdx(inputs[1]);
    return buildOperation(ANEURALNETWORKS_GATHER, {inputIdx, axisIdx, indicesIdx}, getTensorIdxs(outputs));
    */
}

REGISTER_CSINN2_OP_CREATOR(CSINN2Gather, OpType_GatherV2)
} // namespace MNN
