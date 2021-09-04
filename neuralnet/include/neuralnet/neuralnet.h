#ifndef NEURALNET_H
#define NEURALNET_H

#include <vector>

class NeuralNet
{
public:
    NeuralNet(size_t nInputs, const std::vector<size_t>& layerSizes);

    const double* run(const std::vector<double>& inputs);

private:
    size_t nInputs;
    std::vector<size_t> layerSizes;
    std::vector<double> weights;
    std::vector<double> outputs;
};

#endif // NEURALNET_H
