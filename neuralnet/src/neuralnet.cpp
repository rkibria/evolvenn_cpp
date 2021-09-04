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

    maxOutputsSize = *std::max_element(layerSizes.cbegin(), layerSizes.cend());
}

size_t NeuralNet::run(const double* inputs, std::vector<double>& outputs) const
{
    if(outputs.size() < maxOutputsSize) {
        outputs.resize(maxOutputsSize * 2);
    }

    auto inputPtr = inputs;
    auto lastInputs = nInputs;
    size_t weightsBegin = 0;
    size_t outputBegin;
    for(size_t i = 0; i < layerSizes.size(); ++i) {
        outputBegin = (i % 2) * maxOutputsSize;
        const auto lyrSz = layerSizes[i];

        for(size_t n = 0; n < lyrSz; ++n) {
            auto weightedInputs = weights[weightsBegin + n * lyrSz];
            auto weightsIdx = weightsBegin + 1;
            for(size_t w = 0; w < lyrSz; ++w) {
                weightedInputs += weights[weightsBegin + w] * (*inputPtr);
                ++inputPtr;
            }
            outputs[outputBegin + n] = weightedInputs;
        }

        inputPtr = &outputs.data()[outputBegin];
        weightsBegin += 1 + lyrSz * lastInputs;
        lastInputs = lyrSz;
    }

    return outputBegin;
}
