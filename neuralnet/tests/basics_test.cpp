#include "catch.hpp"

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
