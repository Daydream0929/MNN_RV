#ifndef MNN_CSINN2UNARY_HPP
#define MNN_CSINN2UNARY_HPP

#include "CSINN2Backend.hpp"
#include "CSINN2CommonExecution.hpp"

namespace MNN {

class CSINN2Unary : public CSINN2CommonExecution {
public:
    CSINN2Unary(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~CSINN2Unary() = default;
};
} // namespace MNN

#endif // MNN_CSINN2UNARY_HPP
