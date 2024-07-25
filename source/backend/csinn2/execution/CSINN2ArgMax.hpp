#ifndef MNN_CSINN2ARGMAX_HPP
#define MNN_CSINN2ARGMAX_HPP

#include "CSINN2CommonExecution.hpp"
#include "CSINN2Backend.hpp"

namespace MNN {

class CSINN2ArgMax : public CSINN2CommonExecution {
public:
    CSINN2ArgMax(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~CSINN2ArgMax() = default;
};
} // namespace MNN

#endif // MNN_CSINN2ARGMAX_HPP
