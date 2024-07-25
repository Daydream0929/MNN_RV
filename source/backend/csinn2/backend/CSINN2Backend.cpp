#include "CSINN2Backend.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <core/Macro.h>
#include <core/TensorUtils.hpp>
#include <stdlib.h>
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

#ifdef CSINN2_DEBUG
    // #include <android/log.h>
    // #include <sys/time.h>
#endif
namespace MNN {

    void MNNPackC4Uint8(uint8_t* dst, const uint8_t* src, size_t area, size_t depth) {
        int z, x;
        int cur = 0;
        memset(dst, 0, area * UP_DIV(depth, 4) * 4 * sizeof(uint8_t));
        for (z = 0; z < depth; ++z) {
            int plane         = z / 4;
            uint8_t* dstPlane = plane * area * 4 + dst;
            int offset        = z % 4;
            for (x = 0; x < area; ++x) {
                dstPlane[4 * x + offset] = src[cur++];
            }
        }
    }

    void MNNPackC4(float* dst, const float* src, size_t area, size_t depth) {
        int z, x;
        int cur = 0;
        memset(dst, 0, area * UP_DIV(depth, 4) * 4 * sizeof(float));
        for (z = 0; z < depth; ++z) {
            int plane       = z / 4;
            float* dstPlane = plane * area * 4 + dst;
            int offset      = z % 4;
            for (x = 0; x < area; ++x) {
                dstPlane[4 * x + offset] = src[cur++];
            }
        }
    }

    void NHWC2NCHW(const float* source, float* dest, int b, int c, int area) {
        int sourceBatchsize = c * area;
        int destBatchSize   = sourceBatchsize;
        for (int bi = 0; bi < b; ++bi) {
            auto srcBatch = source + bi * sourceBatchsize;
            auto dstBatch = dest + bi * destBatchSize;
            for (int i = 0; i < area; ++i) {
                auto srcArea = srcBatch + i * c;
                auto dstArea = dstBatch + i;
                for (int ci = 0; ci < c; ++ci) {
                    dstArea[ci * area] = srcArea[ci];
                }
            }
        }
    }

    void MNNUnpackC4(float* dst, const float* src, size_t area, size_t depth) {
        int x;
        int z;
        int cur = 0;
        for (z = 0; z < depth; ++z) {
            int plane             = z / 4;
            const float* srcPlane = plane * area * 4 + src;
            int offset            = z % 4;
            for (x = 0; x < area; ++x) {
                dst[cur++] = srcPlane[4 * x + offset];
            }
        }
    }

    void MNNUnpackC4Uint8(uint8_t* dst, const uint8_t* src, size_t area, size_t depth) {
        int x;
        int z;
        int cur = 0;
        for (z = 0; z < depth; ++z) {
            int plane               = z / 4;
            const uint8_t* srcPlane = plane * area * 4 + src;
            int offset              = z % 4;
            for (x = 0; x < area; ++x) {
                dst[cur++] = srcPlane[4 * x + offset];
            }
        }
    }

    void NCHW2NHWC(const float* source, float* dest, int b, int c, int area) {
        int sourceBatchsize = c * area;
        int destBatchSize   = sourceBatchsize;
        for (int bi = 0; bi < b; ++bi) {
            auto srcBatch = source + bi * sourceBatchsize;
            auto dstBatch = dest + bi * destBatchSize;
            for (int i = 0; i < area; ++i) {
                auto srcArea = srcBatch + i;
                auto dstArea = dstBatch + i * c;
                for (int ci = 0; ci < c; ++ci) {
                    dstArea[ci] = srcArea[ci * area];
                }
            }
        }
    }

    ErrorCode tensorConvert(const Tensor* input, const Tensor* output) {
        auto ib     = input->buffer();
        auto ob     = output->buffer();
        auto source = TensorUtils::getDescribe(input)->dimensionFormat;
        auto dest   = TensorUtils::getDescribe(output)->dimensionFormat;
        if (ib.dimensions <= 1 || source == dest) {
            ::memcpy(ob.host, ib.host, input->size());
            return NO_ERROR;
        }
        if (source == MNN_DATA_FORMAT_UNKNOWN || dest == MNN_DATA_FORMAT_UNKNOWN) {
            MNN_ERROR("unknown data format!\nsrc: %s, dst: %s\n", EnumNameMNN_DATA_FORMAT(source), EnumNameMNN_DATA_FORMAT(dest));
            return INVALID_VALUE;
        }
        int area = 1, batch = ib.dim[0].extent, channel;
        if (source == MNN_DATA_FORMAT_NC4HW4 || source == MNN_DATA_FORMAT_NCHW) {
            channel = ib.dim[1].extent;
            for (int axis = 2; axis < ib.dimensions; ++axis) {
                area *= ib.dim[axis].extent;
            }
        } else {
            channel = ib.dim[ib.dimensions - 1].extent;
            for (int axis = 1; axis < ib.dimensions - 1; ++axis) {
                area *= ib.dim[axis].extent;
            }
        }
        const int bitLength = ib.type.bytes();

        if (MNN_DATA_FORMAT_NC4HW4 == source && MNN_DATA_FORMAT_NCHW == dest) {
            if (bitLength == 1) {
                for (int i = 0; i < ib.dim[0].extent; ++i) {
                    MNNUnpackC4Uint8((uint8_t*)ob.host + ob.dim[0].stride * i,
                                    (const uint8_t*)ib.host + ib.dim[0].stride * i, area, channel);
                }
                return NO_ERROR;
            }
            MNN_ASSERT(bitLength == 4);
            for (int i = 0; i < ib.dim[0].extent; ++i) {
                MNNUnpackC4((float*)ob.host + ob.dim[0].stride * i, (const float*)ib.host + ib.dim[0].stride * i, area, channel);
            }
            return NO_ERROR;
        }

        if (MNN_DATA_FORMAT_NCHW == source && MNN_DATA_FORMAT_NC4HW4 == dest) {
            if (bitLength == 1) {
                for (int i = 0; i < ib.dim[0].extent; ++i) {
                    MNNPackC4Uint8((uint8_t*)ob.host + ob.dim[0].stride * i, (const uint8_t*)ib.host + ib.dim[0].stride * i, area, channel);
                }
                return NO_ERROR;
            }
            MNN_ASSERT(bitLength == 4);
            for (int i = 0; i < ib.dim[0].extent; ++i) {
                MNNPackC4((float*)ob.host + ob.dim[0].stride * i, (const float*)ib.host + ib.dim[0].stride * i, area, channel);
            }
            return NO_ERROR;
        }

       if (MNN_DATA_FORMAT_NHWC == source && MNN_DATA_FORMAT_NCHW == dest) {
            if (bitLength != 4) {
                return NOT_SUPPORT;
            }
            NHWC2NCHW((float*)ib.host, (float*)ob.host, batch, channel, area);
        } else if (MNN_DATA_FORMAT_NCHW == source && MNN_DATA_FORMAT_NHWC == dest) {
            if (bitLength != 4) {
                return NOT_SUPPORT;
            }
            NCHW2NHWC((float*)ib.host, (float*)ob.host, batch, channel, area);
        } else {
            return NOT_SUPPORT;
        }

        return NO_ERROR;
    }

    static inline std::map<OpType, CSINN2Backend::Creator*>* getCreatorMap() {
        static std::once_flag of;
        static std::map<OpType, CSINN2Backend::Creator*>* ret = nullptr;
        std::call_once(of, [&]() { ret = new std::map<OpType, CSINN2Backend::Creator*>; });
        return ret;
    }

    bool CSINN2Backend::addCreator(OpType t, Creator* c) {
        auto map = getCreatorMap();
        if (map->find(t) != map->end()) {
            MNN_PRINT("Error: %d type has be added\n", t);
            return false;
        }
        map->insert(std::make_pair(t, c));
        return true;
    }

    CSINN2Backend::CSINN2Backend(const CSINN2Runtime* runtime) : Backend(MNN_FORWARD_USER_0) {
        mCSINN2Runtime = runtime;
        mPrecision  = mCSINN2Runtime->mPrecision;
#ifdef CSINN2DEBUG
        // // Retrieve a handle to libandroid.
        // void *lib = dlopen("libandroid.so", RTLD_NOW || RTLD_LOCAL);
        // // Access the native tracing functions.
        // if (lib != NULL) {
        //     // Use dlsym() to prevent crashes on devices running Android 5.1
        //     // (API level 22) or lower.
        //     ATrace_beginSection = reinterpret_cast<fp_ATrace_beginSection>(
        //         dlsym(lib, "ATrace_beginSection"));
        //     ATrace_endSection = reinterpret_cast<fp_ATrace_endSection>(
        //         dlsym(lib, "ATrace_endSection"));
        //     MNN_PRINT("get function ptr :%p,%p",ATrace_beginSection, ATrace_endSection);
        // }
#endif
    }
    CSINN2Backend::~CSINN2Backend() {

    }

    Execution* CSINN2Backend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) {

        auto map = getCreatorMap();
        auto iter = map->find(op->type());
        
        if (iter == map->end()) {
            MNN_ERROR("map not find !!! \n");
            if(op != nullptr){
                if(op->name() != nullptr){
                    MNN_PRINT("[CSINN2] Don't support type %d, %s\n", op->type(), op->name()->c_str());
                }
            }
            return nullptr;
        }

        auto exe = iter->second->onCreate(inputs, outputs, op, this);

        if (nullptr == exe) {
            MNN_ERROR("nullptr == exe !!! \n");
            if(op != nullptr){
                if(op->name() != nullptr){
                    MNN_PRINT("[CSINN2] The Creator Don't support type %d, %s\n", op->type(), op->name()->c_str());
                }
            }
            return nullptr;
        }

        return exe;
    }

    void CSINN2Backend::CSINN2Backend::onExecuteBegin() const {
    }
    
    void CSINN2Backend::onExecuteEnd() const {
        process();
    }

    Backend::MemObj* CSINN2Backend::onAcquire(const Tensor* tensor, StorageType storageType) {
        bool isInputCopy = TensorUtils::getDescribe(tensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
        bool isOutputCopy = TensorUtils::getDescribe(tensor)->usage==Tensor::InsideDescribe::Usage::OUTPUT;
        if(isInputCopy){
            mInputMap.insert(make_pair((unsigned long)tensor, mInputMap.size()));
        }
        // Don't need extra release
        return new Backend::MemObj;
    }

    bool CSINN2Backend::onClearBuffer() {
        return true;
    }

    void CSINN2Backend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
#ifdef CSINN2DEBUG
        ATrace_beginSection("onCopy");
#endif
        bool isInputCopy = TensorUtils::getDescribe(dstTensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
        bool isOutputCopy = TensorUtils::getDescribe(srcTensor)->usage==Tensor::InsideDescribe::Usage::OUTPUT;
        bool isConst = TensorUtils::getDescribe(srcTensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT || TensorUtils::getDescribe(dstTensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT;
        
        if (isConst) { return; }
        
        if (isInputCopy) {
            const auto iter = mInputIdxMap.find(dstTensor);
            MNN_ASSERT(iter != mInputIdxMap.end());
            memcpy((void*)&mInputTensors[iter->second], &srcTensor, sizeof(void*));
        } else if (isOutputCopy) {
            // MNN_ASSERT(mOutputIdxMap.find(srcTensor) != mOutputIdxMap.end());
            int srcSize = static_cast<int>(TensorUtils::getRawSize(srcTensor) * srcTensor->getType().bytes());
            memcpy(dstTensor->host<void>(), srcTensor->host<void>(), std::min(srcSize, dstTensor->size()));
        }
    
#ifdef CSINN2DEBUG
        ATrace_endSection();
#endif
    }
    
    // CSINN2
    void CSINN2Backend::onResizeBegin() {

    }

    ErrorCode CSINN2Backend::onResizeEnd() {

    }

    int CSINN2Backend::process() const {
        auto ret = modelManager->Run(*(const_cast<vector<shared_ptr<hiai::INDTensorBuffer>>*>(&inputTensors)), *(const_cast<vector<shared_ptr<hiai::INDTensorBuffer>>*>(&outputTensors)));
        return ret;
    }

    CSINN2Runtime::CSINN2Runtime(const Backend::Info& info) {
        mInfo = info;

        BackendConfig::PrecisionMode precision = BackendConfig::Precision_Normal;
        BackendConfig::PowerMode power         = BackendConfig::Power_Normal;
        if (nullptr != mInfo.user) {
            precision = mInfo.user->precision;
            power     = mInfo.user->power;
        }

        mPrecision = precision;
    }

    CSINN2Runtime::~CSINN2Runtime() {}

    Backend* CSINN2Runtime::onCreate(const BackendConfig* config) const {
        return new CSINN2Backend(this);
    }

    void CSINN2Runtime::onGabageCollect(int level) {
        // nothing now
    }
    Runtime::CompilerType CSINN2Runtime::onGetCompilerType() const {
        return Compiler_Origin;
    }

    struct CSINN2BackendCreator : RuntimeCreator {

        virtual Runtime* onCreate(const Backend::Info& info) const override {
            return new CSINN2Runtime(info);
        }

        virtual bool onValid(Backend::Info& info) const override {
            return true;
        }
    };

    void registerCSINN2RuntimeCreator() {

    }
}
