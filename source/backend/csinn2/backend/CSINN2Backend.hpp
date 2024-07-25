#ifndef MNN_CSINN2BACKEND_H
#define MNN_CSINN2BACKEND_H

#include <core/Backend.hpp>
#include <core/Execution.hpp>
#include "MNN_generated.h"

namespace MNN {
    
class CSINN2Runtime : public Runtime {
public:
    CSINN2Runtime(const Backend::Info& info);
	virtual ~CSINN2Runtime();
	virtual CompilerType onGetCompilerType() const override;
	virtual Backend* onCreate(const BackendConfig* conf) const override;
	virtual void onGabageCollect(int level) override;
	virtual std::pair<const void*, size_t> onGetCache() override {
	    return std::make_pair(mCacheBuffer, mCacheSize);
	}	

private:
	Backend::Info mInfo;
	BackendConfig::PrecisionMode mPrecision;
	const void* mCacheBuffer = nullptr;
	size_t mCacheSize = 0;

	friend class CSINN2Backend;
};

class CSINN2Backend : public Backend {
public:
 	CSINN2Backend(const CSINN2Runtime* runtime);
	virtual ~CSINN2Backend();

	virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) override;

    virtual void onExecuteBegin() const override;
    virtual void onExecuteEnd() const override;

    virtual Backend::MemObj* onAcquire(const Tensor* tensor, StorageType storageType) override;
    virtual bool onClearBuffer() override;
    virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override;

    virtual void onResizeBegin() override;
    virtual ErrorCode onResizeEnd() override;

public:
    class Creator {
    public:
        virtual Execution* onCreate(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs,
                                    const MNN::Op* op, Backend* backend) const = 0;
    };
    static bool addCreator(OpType t, Creator* c);
    // CSINN2 functions
    private:
        std::vector<std::string> mModelName;
        const CSINN2Runtime* mNPURuntime;
        BackendConfig::PrecisionMode mPrecision;
        std::vector<std::unique_ptr<Tensor>> mInputContentTensors, mOutputContentTensors;
        std::vector<const Tensor*> mInputTensors, mOutputTensors;
        // tensor idx map
        std::map<const Tensor*, uint32_t> mTensorIdxMap, mInputIdxMap, mOutputIdxMap, mDequantIdxMap;
        std::map<uint32_t, const Tensor*> mDequantMap;
        std::map<uint32_t, uint32_t> mQuantCacheMap;
        uint32_t mTensorIdx = 0;
        std::vector<const char*> mOpNames;
        // scalar idx map
        std::map<int, uint32_t> mScalarIntMap;
        std::map<bool, uint32_t> mScalarBoolMap;
        std::map<float, uint32_t> mScalarFloatMap;
        // fp16 buffer
        std::vector<std::unique_ptr<int16_t[]>> mHalfBuffer;
        // CSINN2 resource
    

};

template <class T>
class CSINN2CreatorRegister {
public: 
    CSINN2CreatorRegister(OpType type) {
        T *t = new T;
        CSINN2Backend::addCreator(type, t);
    }
    ~CSINN2CreatorRegister() = default;
};

template <typename T>
class TypedCreator : public CSINN2Backend::Creator {
public:
    virtual ~TypedCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op,
                                Backend *backend) const override {
        auto newOp = new T(backend, op, inputs, outputs);
        return newOp;
    }
};

#define REGISTER_CSINN2_OP_CREATOR(name, opType)     \
    void ___##name##__##opType##__() {            \
        static TypedCreator<name> _temp;\
        CSINN2Backend::addCreator(opType, &_temp); \
    }

}


#endif  // MNN_CSINN2BACKEND_H
