#include <catch2/catch.hpp>

#include "neuralnet/neuralnet.h"

TEST_CASE( "Neural net weights have expected number", "[neuralnet]" ) {
    NeuralNet nn(2, {2, 3, 2});
    REQUIRE(nn.getWeights().size() == 23);
}

TEST_CASE( "Zeroed weights yields zeroed outputs", "[neuralnet]" ) {
    NeuralNet nn(2, {2, 3, 2});
    nn.setWeights({   0, 0, 0,
                      0, 0, 0,

                      0, 0, 0,
                      0, 0, 0,
                      0, 0, 0,

                      0, 0, 0, 0,
                      0, 0, 0, 0
                  });
    std::vector<double> outputs;
    const double inputs[2] = {1.0, 2.0};
    const auto result = nn.run(inputs, outputs);
    REQUIRE(result == 0);
    REQUIRE(outputs[0] == 0);
    REQUIRE(outputs[1] == 0);
}

TEST_CASE( "Correct output if only bias weights are set", "[neuralnet]" ) {
    NeuralNet nn(2, {2, 3, 2});
    nn.setWeights({   1, 0, 0,
                      2, 0, 0,

                      3, 0, 0,
                      4, 0, 0,
                      5, 0, 0,

                      1234, 0, 0, 0,
                      5678, 0, 0, 0
                  });
    std::vector<double> outputs;
    const double inputs[2] = {1.0, 2.0};
    const auto result = nn.run(inputs, outputs);
    REQUIRE(outputs[0] == 1234);
    REQUIRE(outputs[1] == 5678);
}

TEST_CASE( "Each neuron sums up each input one to one", "[neuralnet]" ) {
    NeuralNet nn(2, {2, 3, 2});
    nn.setWeights({   0, 1, 1,
                      0, 1, 1,

                      0, 1, 1,
                      0, 1, 1,
                      0, 1, 1,

                      0, 1, 1, 1,
                      0, 1, 1, 1
                  });
    std::vector<double> outputs;
    const double inputs[2] = {1.0, 2.0};
    const auto result = nn.run(inputs, outputs);
    REQUIRE(outputs[0] == 18);
    REQUIRE(outputs[1] == 18);
}

TEST_CASE( "Activation is applied correctly", "[neuralnet]" ) {
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
    const double inputs[2] = {1.0, 2.0};
    const auto result = nn.run(inputs, outputs);
    REQUIRE(outputs[0] == 6);
    REQUIRE(outputs[1] == 0);
}
