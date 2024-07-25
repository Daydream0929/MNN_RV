#include "CSINN2Scale.hpp"

namespace MNN {


CSINN2Scale::CSINN2Scale(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : CSINN2CommonExecution(b, op) {
}

ErrorCode CSINN2Scale::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto scaleParam = mOp->main_as_Scale();
    auto weight = scaleParam->scaleData();
    auto bias = scaleParam->biasData();
    uint32_t channel = scaleParam->channels();    
    auto inputShape = inputs[0]->shape(); // NCHW
    int inputSize = inputs[0]->elementSize();
    std::vector<uint32_t> dims {1, 1, 1, channel};
    // middle = input * weight

    /* CSINN2
    auto middleIdx = buildTensor(ANEURALNETWORKS_TENSOR_FLOAT32, inputShape);
    auto inputIdxs = getTensorIdxs(inputs);
    inputIdxs.push_back(buildConstant(weight->data(), channel * sizeof(float), ANEURALNETWORKS_TENSOR_FLOAT32, dims));
    inputIdxs.push_back(buildScalar(ANEURALNETWORKS_FUSED_NONE));
    buildOperation(ANEURALNETWORKS_MUL, inputIdxs, {middleIdx});
    // output = middle + bias
    auto biasIdx = buildConstant(bias->data(), channel * sizeof(float), ANEURALNETWORKS_TENSOR_FLOAT32, dims);
    return buildOperation(ANEURALNETWORKS_ADD, {middleIdx, biasIdx, buildScalar(ANEURALNETWORKS_FUSED_NONE)}, getTensorIdxs(outputs));
    */
}

REGISTER_CSINN2_OP_CREATOR(CSINN2Scale, OpType_Scale)
} // namespace MNN
