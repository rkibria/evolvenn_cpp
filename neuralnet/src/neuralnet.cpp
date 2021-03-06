#include "neuralnet/neuralnet.h"

#include <cassert>
#include <algorithm>

NeuralNet::NeuralNet(size_t nInputs_, const std::vector<size_t>& layerSizes_, bool outputIsLinear)
    : nInputs{ nInputs_ },
      layerSizes{ layerSizes_ },
      outputLinear{ outputIsLinear }
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

        const bool isOutputLayer = (lrIdx == layerSizes.size() - 1);
        const bool linearOutput = isOutputLayer && outputLinear;
        for(size_t neuIdx = 0; neuIdx < lrSz; ++neuIdx) {
            auto weightedInputs = weights[weightsBegin++];
            for(size_t w = 0; w < lastInputs; ++w) {
                weightedInputs += weights[weightsBegin++] * (*(inputPtr + w));
            }
            // https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
            const auto activation = linearOutput ? weightedInputs : std::max(0.0, weightedInputs);
            outputs[outputBegin + neuIdx] = activation;
        }

        inputPtr = &outputs.data()[outputBegin];
        lastInputs = lrSz;
    }

    return outputBegin;
}
