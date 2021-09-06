#include <iostream>
#include <cmath>
#include <chrono>
#include <random>
#include <iomanip>
#include <cassert>

#include "neuralnet/neuralnet.h"
#include "population/population.h"

void speedTest1()
{
    NeuralNet nn(2, {2, 3, 2});
    nn.setWeights({   0, 1, 1,
                      0, 1, 1,

                      0, 1, 1,
                      0, 1, 1,
                      0, 1, 1,

                      0, 1, 1, -1,
                      0, 1, -1, -1
                  });
    std::vector<double> outputs;
    double inputs[2] = {1.0, 2.0};

    const auto start = std::chrono::high_resolution_clock::now();

    for(size_t i = 0; i < 1000000; ++i) {
        nn.run(inputs, outputs);
        inputs[0] += 0.00001;
    }

    const auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "Time 1: " << duration.count() << " ms" << std::endl;
}

std::default_random_engine generator;

class NnIndividual : public Individual
{
public:
    double getVariation(double spread)
    {
        const double mean = 0.0;
        const double stddev = spread;
        std::normal_distribution<double> dist(mean, stddev);
        return spread * (dist(generator) - spread/2);
    }

    NnIndividual() : nn(2, {2, 2, 1})
    {
        auto& weights = nn.getWeights();
        for(auto& w : weights) {
            w = getVariation(1.0);
        }
    }

    ~NnIndividual() override
    {
    }

    void evaluate() override
    {
        std::vector<double> outputs;
        double inputsArray[2];
        auto inputs = &inputsArray[0];

        inputs[0] = -1;
        inputs[1] = -1;
        auto resultIdx = nn.run(inputs, outputs);
        fitness += (outputs[resultIdx] < 0.5) ? 0 : 1;

        inputs[0] = -1;
        inputs[1] = 1;
        resultIdx = nn.run(inputs, outputs);
        fitness += (outputs[resultIdx] > 0.5) ? 0 : 1;

        inputs[0] = 1;
        inputs[1] = -1;
        resultIdx = nn.run(inputs, outputs);
        fitness += (outputs[resultIdx] > 0.5) ? 0 : 1;

        inputs[0] = 1;
        inputs[1] = 1;
        resultIdx = nn.run(inputs, outputs);
        fitness += (outputs[resultIdx] < 0.5) ? 0 : 1;


//        for(int i = 0; i < 100; ++i) {
//            const double x = -M_PI + 2 * M_PI / 100 * i;
//            double inputs = x * 1.0;
//            const auto resultIdx = nn.run(&inputs, outputs);
//            double val = 0.0;
//            int inc = 1;
//            for(size_t k =  0; k < 8; ++k) {
//                const auto result = outputs[resultIdx + k];
//                const bool doInc = (result > 10);
//                val += inc * (doInc ? 1 : 0);
//                inc *= 2;
//            }
//            val = (val - 127) / 255;
//            val = outputs[resultIdx + 0];
//            const auto expect = sin(x);
//            auto diff = fabs(val - expect);
//            diff *= diff;
//            std::cout << "\nval " << val << " vs real " << expect << " diff " << diff << "\n";
//            std::cout << diff << " ";
//            fitness += diff;
//        }
//        std::cout << "idv fit " <<  std::setprecision(30) << fitness << "\n";
    }

    void mutate(double spread) override
    {
        auto& weights = nn.getWeights();
        for(auto& w : weights) {
            const auto variation = getVariation(spread);
//            std::cout << w << " mutate " << variation << " // spread " << spread << "\n";
            w += variation;
        }
    }

    void mutateFrom(const Individual* other, double spread) override
    {
        const auto otherNn = dynamic_cast<const NnIndividual*>(other);
        assert(this->nn.getInputs() == otherNn->nn.getInputs());
        assert(this->nn.getLayerSizes() == otherNn->nn.getLayerSizes());

        const auto& otherWeights = otherNn->nn.getWeights();
        auto& weights = nn.getWeights();
        for(size_t i = 0; i < weights.size(); ++i) {
            const auto variation = getVariation(spread);
//            std::cout << otherWeights[i] << " mutateFrom " << variation << " // spread " << spread << "\n";
            weights[i] = otherWeights[i] + variation;
        }
    }

    void dump(std::ostream& os) const override
    {
        os << nn.getWeights()[0] << "/" << nn.getWeights()[1] << "/" << nn.getWeights()[2];
    }

    NeuralNet nn;
};

void evolution1()
{
    const size_t popSize = 1000;
    const double mutationSpread = 1;
    const double mutationHalflife = 100;
    const size_t numGens = 1000;

    Population pop;
    for(size_t i = 0; i < popSize; ++i) {
        pop.addIndividual(std::make_unique<NnIndividual>());
    }

    const auto start = std::chrono::high_resolution_clock::now();

    size_t generation = 1;
    while(generation < numGens) {
        const double halflifeFactor = 1.0; // TODO this returns inf (mutationHalflife > 0) ? pow(2.0, -(generation % 1000) / mutationHalflife) : 1;
        const double spread = (1.0 - (generation / static_cast<double>(numGens))) * 0.25; // TODO std::max(mutationSpread * halflifeFactor, mutationSpread * 0.01);

        pop.evolve(spread);

        if(generation % 1 == 0) {
            double sum = 0;
            for(size_t i = 0; i < pop.size(); ++i) {
//                std::cout << i << " fitness " << pop.getIndividual(i)->getFitness() << "\n";
                sum += pop.getIndividual(i)->getFitness();
            }
            std::cout << "gen " << generation << ": " << pop.getIndividual(0)->getFitness()
                      << " // spread " << spread
                      << " // avg " << sum / pop.size() << "\n";
        }

        ++generation;
    }

    const auto stop = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Time: " << duration.count() << " ms" << std::endl;
}

int main(int argc, char **argv)
{
//    speedTest1();
    evolution1();

    return 0;
}
