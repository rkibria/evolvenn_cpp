#include "catch.hpp"

#include "neuralnet/neuralnet.h"

TEST_CASE( "Can create a neural net object", "[neuralnet]" ) {
    NeuralNet nn(4, {4, 4, 2});
    REQUIRE( true );
}
