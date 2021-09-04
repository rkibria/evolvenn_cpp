#include "catch.hpp"

#include "neuralnet/neuralnet.h"

TEST_CASE( "Neural net weights have expected number", "[neuralnet]" ) {
    NeuralNet nn(2, {2, 3, 2});
    REQUIRE( nn.getWeights().size() == 23 );
}
