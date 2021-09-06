#ifndef NEURALNET_H
#define NEURALNET_H

#include <vector>

class NeuralNet
{
public:
    NeuralNet(size_t nInputs, const std::vector<size_t>& layerSizes);

    size_t run(const double* inputs, std::vector<double>& outputs) const;

    std::vector<double>& getWeights() { return weights; }
    const std::vector<double>& getWeights() const { return weights; }
    void setWeights(std::vector<double>&& w) { weights = w; }
    void setWeights(const std::vector<double>& w) { weights = w; }

    size_t getInputs() const { return nInputs; }
    const std::vector<size_t>& getLayerSizes() const { return layerSizes; }

private:
    size_t nInputs;
    std::vector<size_t> layerSizes;
    std::vector<double> weights;
};

#endif // NEURALNET_H
