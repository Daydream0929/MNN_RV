#ifndef MNN_CSINN2CONVOLUTION_HPP
#define MNN_CSINN2CONVOLUTION_HPP

#include "CSINN2Backend.hpp"
#include "CSINN2CommonExecution.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {

class CSINN2Convolution : public CSINN2CommonExecution {
public:
    CSINN2Convolution(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~CSINN2Convolution() = default;
private:
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    std::unique_ptr<float[]> nhwcWeight;
    std::unique_ptr<int8_t[]> quantWeight;
    std::unique_ptr<int32_t[]> quantBias;
    bool isDepthwise = false, isDeconv = false;
};
} // namespace MNN

#endif // MNN_CSINN2CONVOLUTION_HPP
