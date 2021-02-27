//
//  Optimizer.cpp
//  MNN
//
//  Created by MNN on 2019/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Optimizer.hpp>
#include "MergeOptimizer.hpp"
#include "core/Backend.hpp"
namespace MNN {
namespace Express {
Optimizer::Parameters::Parameters(int n) {
    MNN_ASSERT(n > 0);
    mValue = new float[n];
    mSize  = n;
}
Optimizer::Parameters::~Parameters() {
    if (nullptr != mValue) {
        delete[] mValue;
    }
}
float* Optimizer::Parameters::get() const {
    return mValue;
}
int Optimizer::Parameters::size() const {
    return mSize;
}

std::shared_ptr<Optimizer::Parameters> Optimizer::onGetParameters(const std::vector<VARP>& outputs) {
    return nullptr;
}

std::shared_ptr<Optimizer> Optimizer::create(Config config) {
    // Do nothing
    return nullptr;
}

} // namespace Express
} // namespace MNN
