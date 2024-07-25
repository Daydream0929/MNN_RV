#ifndef MNN_CSINN2POOL_HPP
#define MNN_CSINN2POOL_HPP

#include "CSINN2CommonExecution.hpp"
#include "CSINN2Backend.hpp"

namespace MNN {

class CSINN2Pool : public CSINN2CommonExecution {
public:
    CSINN2Pool(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~CSINN2Pool() = default;
private:
    void addPadLayer(const Tensor * input, const Pool* common);
    std::string mPoolInputName, mPoolOutputName;
};
} // namespace MNN

#endif // MNN_CSINN2POOL_HPP
