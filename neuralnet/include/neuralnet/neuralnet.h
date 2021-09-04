#ifndef NEURALNET_H
#define NEURALNET_H

#include <vector>

class NeuralNet
{
public:
    NeuralNet(int nInputs, const std::vector<int>& layerSizes);

private:
    std::vector<double> layerSizes;
    std::vector<double> weights;
    std::vector<double> outputs;
};

#endif // NEURALNET_H
