#ifndef MNN_CSINN2INTERP_HPP
#define MNN_CSINN2INTERP_HPP

#include "CSINN2Backend.hpp"
#include "CSINN2CommonExecution.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {

class CSINN2Interp : public CSINN2CommonExecution {
public:
    CSINN2Interp(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~CSINN2Interp() = default;
};
} // namespace MNN

#endif // MNN_CSINN2INTERP_HPP
