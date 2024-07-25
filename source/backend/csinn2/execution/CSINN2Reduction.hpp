#ifndef MNN_CSINN2REDUCTION_HPP
#define MNN_CSINN2REDUCTION_HPP

#include "CSINN2Backend.hpp"
#include "CSINN2CommonExecution.hpp"

namespace MNN {

class CSINN2Reduction : public CSINN2CommonExecution {
public:
    CSINN2Reduction(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~CSINN2Reduction() = default;
};
} // namespace MNN

#endif // MNN_CSINN2REDUCTION_HPP
