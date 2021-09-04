#include "neuralnet/neuralnet.h"

#include <cassert>
#include <algorithm>

NeuralNet::NeuralNet(size_t nInputs_, const std::vector<size_t>& layerSizes_)
    : nInputs{ nInputs_ },
      layerSizes{ layerSizes_ }
{
    size_t nWeights = 0;
    auto lastInputs = nInputs;
    for(size_t i = 0; i < layerSizes.size(); ++i) {
        const auto lyrSz = layerSizes[i];
        nWeights += (1 + lyrSz) * lastInputs;
        lastInputs = lyrSz;
    }
    weights.resize(nWeights);

    const auto maxItr = std::max_element(layerSizes.cbegin(), layerSizes.cend());
    if(maxItr != layerSizes.cend()) {
        outputs.resize(*maxItr * 2);
    }
}

const double* NeuralNet::run(const double* inputs)
{
    const auto maxOutputsSize = outputs.size() / 2;
    auto inputPtr = inputs;
    auto lastInputs = nInputs;
    size_t weightsBegin = 0;
    size_t outputStart = 0;
    for(size_t i = 0; i < layerSizes.size(); ++i) {
        outputStart = (i % 2) * maxOutputsSize;
        const auto lyrSz = layerSizes[i];
        std::fill(outputs.begin() + outputStart, outputs.end() + outputStart + lyrSz, 0);

        for(size_t n = 0; n < lyrSz; ++n) {
            auto weightedInputs = weights[weightsBegin + n * lyrSz];
            auto weightsIdx = weightsBegin + 1;
            for(size_t w = 0; w < lyrSz; ++w) {
                weightedInputs += weights[weightsBegin + w] * (*inputPtr);
                ++inputPtr;
            }
            outputs[outputStart + n] = weightedInputs;
        }

        inputPtr = &outputs.data()[outputStart];
        weightsBegin += 1 + lyrSz * lastInputs;
        lastInputs = lyrSz;
    }

    return &outputs.data()[outputStart];
}
