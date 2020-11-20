//
//  GeometryReduce.cpp
//  MNN
//
//  Created by MNN on 2020/06/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "core/OpCommonUtils.hpp"
namespace MNN {
class GeometryReduce : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        MNN_ASSERT(1 == outputs.size());
        MNN_ASSERT(inputs.size() >= 1);
        auto reduceDims      = OpCommonUtils::computeReduceDims(inputs, op);
        auto reduct          = op->main_as_ReductionParam();
        auto reductOp        = reduct->operation();
        Tensor* currentInput = inputs[0];
        MNN_ASSERT(reduceDims.size() > 0);
        for (int i = 0; i < reduceDims.size(); ++i) {
            auto& iter   = reduceDims[i];
            auto inside  = std::get<2>(iter);
            auto outside = std::get<0>(iter);
            auto axis    = std::get<1>(iter);
            
            std::shared_ptr<Tensor> inputTensor(
                Tensor::createDevice({outside, axis, inside}, inputs[0]->getType()));
            auto des        = TensorUtils::getDescribe(inputTensor.get());
            des->memoryType = Tensor::InsideDescribe::MemoryType::MEMORY_VIRTUAL;
            des->regions    = {TensorUtils::makeFullSlice(currentInput)};
            res.extras.emplace_back(inputTensor);
            std::shared_ptr<Tensor> outputTensor(
                Tensor::createDevice({outside, 1, inside}, inputs[0]->getType()));
            res.extras.emplace_back(outputTensor);

            // Create Command
            {
                std::unique_ptr<OpT> sum(new OpT);
                sum->type                               = OpType_Reduction;
                sum->main.type                          = OpParameter_ReductionParam;
                sum->main.value                         = new ReductionParamT;
                sum->main.AsReductionParam()->dim       = {1};
                sum->main.AsReductionParam()->keepDims  = true;
                sum->main.AsReductionParam()->operation = reductOp;
                flatbuffers::FlatBufferBuilder builder;
                auto lastOffset = Op::Pack(builder, sum.get());
                builder.Finish(lastOffset);
                Command cmd;
                cmd.buffer.resize(builder.GetSize());
                ::memcpy(cmd.buffer.data(), builder.GetBufferPointer(), cmd.buffer.size());
                cmd.inputs  = {inputTensor.get()};
                cmd.outputs = {outputTensor.get()};
                cmd.op      = flatbuffers::GetMutableRoot<Op>(cmd.buffer.data());
                res.command.emplace_back(std::move(cmd));
            }
            currentInput = outputTensor.get();
            // Ref output
            if (i == reduceDims.size() - 1) {
                auto outputDes        = TensorUtils::getDescribe(outputs[0]);
                outputDes->memoryType = Tensor::InsideDescribe::MemoryType::MEMORY_VIRTUAL;
                outputDes->regions    = {TensorUtils::makeFullSlice(outputTensor.get())};
            }
        }
        return true;
    }
};
static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryReduce);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Reduction});
}

REGISTER_GEOMETRY(GeometryReduce, _create);

} // namespace MNN
