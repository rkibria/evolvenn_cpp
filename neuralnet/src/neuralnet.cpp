#include "neuralnet/neuralnet.h"

#include <cassert>
#include <algorithm>

NeuralNet::NeuralNet(size_t nInputs_, const std::vector<size_t>& layerSizes_)
    : nInputs{ nInputs_ }
{
    size_t nWeights = 0;
    auto lastInputs = nInputs;
    for(size_t i = 0; i < layerSizes.size(); ++i) {
        const auto lrSz = layerSizes[i];
        nWeights += (1 + lrSz) * lastInputs;
        lastInputs = lrSz;
    }
    weights.resize(nWeights);
}

size_t NeuralNet::run(const double* inputs, std::vector<double>& outputs) const
{
    const auto maxOutputsSize = *std::max_element(layerSizes.cbegin(), layerSizes.cend());
    if(outputs.size() < maxOutputsSize) {
        outputs.resize(maxOutputsSize * 2);
    }

    auto inputPtr = inputs;
    auto lastInputs = nInputs;
    size_t weightsBegin = 0;
    size_t outputBegin = 0;
    for(size_t lrIdx = 0; lrIdx < layerSizes.size(); ++lrIdx) {
        const auto lrSz = layerSizes[lrIdx];
        outputBegin = (lrIdx % 2 ) ? maxOutputsSize : 0;

        for(size_t neuIdx = 0; neuIdx < lrSz; ++neuIdx) {
            auto weightedInputs = weights[weightsBegin++];
            for(size_t w = 0; w < lastInputs; ++w) {
                weightedInputs += weights[weightsBegin++] * (*(inputPtr + w));
            }
            // https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
            const auto activation = std::max(0.0, weightedInputs);
            outputs[outputBegin + neuIdx] = activation;
        }

        inputPtr = &outputs.data()[outputBegin];
        lastInputs = lrSz;
    }

    return outputBegin;
}

NeuralNet2::NeuralNet2(size_t nInputs_, const std::vector<size_t>& layerSizes_)
    : nInputs{ nInputs_ },
      weights( layerSizes_.size() )
{
    for(size_t i = 0; i < layerSizes_.size(); ++i) {
        weights[i].resize(layerSizes_[i] + 1);
    }
}

size_t NeuralNet2::run(const double* inputs, std::vector<double>& outputs) const
{
    size_t maxOutputsSize = 0;
    for(const auto& curWeights : weights) {
        maxOutputsSize = std::max(maxOutputsSize, curWeights.size() - 1);
    }
    if(outputs.size() < maxOutputsSize) {
        outputs.resize(maxOutputsSize * 2);
    }

    auto inputPtr = inputs;
    size_t outputBegin = 0;

    for(size_t lrIdx = 0; lrIdx < weights.size(); ++lrIdx) {
        const auto& curWeights = weights[lrIdx];
        const auto lrSz = curWeights.size() - 1;
        outputBegin = (lrIdx % 2 ) ? maxOutputsSize : 0;

        for(size_t neuIdx = 0; neuIdx < lrSz; ++neuIdx) {
            auto weightedInputs = curWeights[0];
            for(size_t w = 1; w < curWeights.size(); ++w) {
                weightedInputs += curWeights[w] * (*(inputPtr + (w - 1)));
            }
            // https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
            const auto activation = std::max(0.0, weightedInputs);
            outputs[outputBegin + neuIdx] = activation;
        }

        inputPtr = &outputs.data()[outputBegin];
    }

    return outputBegin;
}
