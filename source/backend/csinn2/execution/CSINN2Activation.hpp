#ifndef MNN_CSINN2ACTIVATION_HPP
#define MNN_CSINN2ACTIVATION_HPP

#include "CSINN2CommonExecution.hpp"
#include "CSINN2Backend.hpp"

namespace MNN {

class CSINN2Activation : public CSINN2CommonExecution {
public:
    CSINN2Activation(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~CSINN2Activation() = default;
};
} // namespace MNN

#endif // MNN_CSINN2ACTIVATION_HPP
