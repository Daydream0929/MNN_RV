#ifndef MNN_CSINN2QUANT_HPP
#define MNN_CSINN2QUANT_HPP

#include "CSINN2Backend.hpp"
#include "CSINN2CommonExecution.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {

class CSINN2Quant : public CSINN2CommonExecution {
public:
    CSINN2Quant(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~CSINN2Quant() = default;
};

class CSINN2Dequant : public CSINN2CommonExecution {
public:
    CSINN2Dequant(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~CSINN2Dequant() = default;
};
} // namespace MNN

#endif // MNN_CSINN2QUANT_HPP
