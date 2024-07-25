#include "CSINN2CommonExecution.hpp"

namespace MNN {

CSINN2CommonExecution::CSINN2CommonExecution(Backend *backend, const Op *op) : Execution(backend), mOp(op) {
    mCSINN2Backend = (CSINN2Backend *)backend;
}

ErrorCode CSINN2CommonExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    return NO_ERROR;
}

ErrorCode CSINN2CommonExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    return NO_ERROR;
}

std::vector<uint32_t> CSINN2CommonExecution::getTensorIdxs(const std::vector<Tensor*>& tensors) {
    std::vector<uint32_t> idxs(tensors.size());
    for (int i = 0; i < tensors.size(); i++) {
        idxs[i] = mCSINN2Backend->getTensorIdx(tensors[i], true);
    }
    return idxs;
}

}; // namespace MNN
