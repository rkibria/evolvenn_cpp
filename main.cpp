#include <iostream>
//#include <vector>
#include <cmath>
#include <chrono>

//#include <tbb/parallel_for.h>

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
        outputs[0] += 0.00001;
    }

    const auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "Time 1: " << duration.count() << " ms" << std::endl;
}

class NnIndividual : public Individual
{
public:
    NnIndividual() : nn(1, {4, 4, 4})
    {
    }

    ~NnIndividual() override
    {
    }

    void mutate(double spread) override
    {

    }

    void evaluate() override
    {

    }

    NeuralNet nn;
};

void evolution1()
{
    const size_t popSize = 1000;
    const double mutationSpread = 25;
    const size_t mutationHalflife = 100;

    Population pop;
    for(size_t i = 0; i < popSize; ++i) {
        pop.addIndividual(std::make_unique<NnIndividual>());
    }

    const auto start = std::chrono::high_resolution_clock::now();

    size_t generation = 1;
    while(generation < 100) {
        const double halflifeFactor = (mutationHalflife > 0) ? pow(2, -(generation % 1000) / mutationHalflife) : 1;
        const double spread = std::max(mutationSpread * halflifeFactor, mutationSpread * 0.01);

        pop.evolve(spread);
        std::cout << "gen " << generation << ": " << pop.getIndividual(0)->getFitness() << "\n";

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
