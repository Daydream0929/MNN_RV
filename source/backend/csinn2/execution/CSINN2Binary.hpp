#ifndef MNN_CSINN2BIANRY_HPP
#define MNN_CSINN2BIANRY_HPP

#include "CSINN2CommonExecution.hpp"
#include "CSINN2Backend.hpp"

namespace MNN {

class CSINN2Binary : public CSINN2CommonExecution {
public:
    CSINN2Binary(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~CSINN2Binary() = default;
};
} // namespace MNN

#endif // MNN_CSINN2BINARY_HPP
