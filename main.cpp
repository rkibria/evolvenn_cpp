#include <iostream>
//#include <vector>
//#include <cmath>
#include <chrono>

//#include <tbb/parallel_for.h>

#include "neuralnet/neuralnet.h"

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

int main(int argc, char **argv)
{
    speedTest1();

    return 0;
}
