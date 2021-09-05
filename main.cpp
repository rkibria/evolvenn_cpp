#include <iostream>
#include <cmath>
#include <chrono>
#include <random>

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

class NnIndividual : public Individual
{
public:
    NnIndividual() : nn(1, {8, 8, 8})
    {
        const double mean = 0.0;
        const double stddev = 1.0;
        std::default_random_engine generator;
        std::normal_distribution<double> dist(mean, stddev);

        auto& weights = nn.getWeights();
        for(auto& w : weights) {
            w = dist(generator);
        }
    }

    ~NnIndividual() override
    {
    }

    void mutate(double spread) override
    {
        const double mean = 0.0;
        const double stddev = 1.0;
        std::default_random_engine generator;
        std::normal_distribution<double> dist(mean, stddev);

        auto& weights = nn.getWeights();
        for(auto& w : weights) {
            w += spread * (dist(generator) - 0.5);
        }
    }

    void evaluate() override
    {
        std::vector<double> outputs;
        for(int i = 0; i < 100; ++i) {
            const double x = -M_PI + 2 * M_PI / 100 * i;
            double inputs = x;
            const auto resultIdx = nn.run(&inputs, outputs);
            double val = 0.0;
            int inc = 1;
            for(size_t k =  0; k < 8; ++k) {
                const auto result = outputs[resultIdx + k];
                val += inc * (result > 1 ? 1 : 0);
                inc *= 2;
            }
            val = (val - 127) / 255;
            auto diff = val - sin(x);
            diff *= diff;
            fitness += diff;
        }
    }

    void copyFrom(const Individual* other) override
    {
        nn = dynamic_cast<const NnIndividual*>(other)->nn;
    }

    NeuralNet nn;
};

void evolution1()
{
    const size_t popSize = 10;
    const double mutationSpread = 1;
    const double mutationHalflife = 100;

    Population pop;
    for(size_t i = 0; i < popSize; ++i) {
        pop.addIndividual(std::make_unique<NnIndividual>());
    }

    const auto start = std::chrono::high_resolution_clock::now();

    size_t generation = 1;
    while(generation < 1000) {
        const double halflifeFactor = (mutationHalflife > 0) ? pow(2, -(generation % 1000) / mutationHalflife) : 1;
        const double spread = std::max(mutationSpread * halflifeFactor, mutationSpread * 0.01);

        pop.evolve(spread);
        std::cout << "gen " << generation << ": " << pop.getIndividual(0)->getFitness() << "\n";

        double sum = 0;
        for(size_t i = 0; i < pop.size(); ++i) {
            std::cout << i << " fitness " << pop.getIndividual(i)->getFitness() << "\n";
            sum += pop.getIndividual(i)->getFitness();
        }
        std::cout << "avg " << sum << "\n";
        int zz;
        std::cin >> zz;

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
