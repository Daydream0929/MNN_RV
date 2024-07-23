#ifndef MNN_CSINN2COMMONEXECUTION_HPP
#define MNN_CSINN2COMMONEXECUTION_HPP

#include "core/Execution.hpp"
#include "CSINN2Backend.hpp"
#include <memory>

namespace MNN {
    
class CSINN2CommonExecution : public Execution {
public:
    CSINN2CommonExecution(Backend *backend, const Op *op);
    virtual ~CSINN2CommonExecution() = default;
    
    virtual ErrorCode onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
protected:
    bool mNCHW;
    CSINN2Backend* mCSINN2Backend;
    const Op* mop;
};

} // namespace MNN

#endif // MNN_CSINN2COMMONEXECUTION_HPP


