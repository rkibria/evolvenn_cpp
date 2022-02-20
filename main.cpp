#include <iostream>
#include <cmath>
#include <chrono>
#include <random>
#include <iomanip>
#include <cassert>

#include "neuralnet/neuralnet.h"
#include "population/population.h"

#include "htmlanim_shapes.hpp"

std::default_random_engine generator;

constexpr int sections = 10;

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

    NnIndividual() : nn(1, {2, 2, 1}, true)
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

        // Map: [-PI,PI] -> [-1,1]
        for(int i = 0; i < sections + 1; ++i) {
            const double x = -M_PI + 2 * M_PI / sections * i;
            const auto expect = sin(x);

            double inputs = x / M_PI;

            const auto resultIdx = nn.run(&inputs, outputs);

            const auto actual = outputs[resultIdx];

            auto diff = fabs(actual - expect);
            const auto diffsq = diff * diff;
            fitness += diffsq;
            // std::cout << i << ") x " << x << " exp " << expect << " act " << actual 
            //     << " dsq " << diffsq << " fit " << fitness;
            // std::cout << "\n";
        }
    }

    void mutate(double spread) override
    {
        auto& weights = nn.getWeights();
        for(auto& w : weights) {
            const auto variation = getVariation(spread);
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
    HtmlAnim::HtmlAnim anim("sinus evolution progress");
    const int outW = 500, outH = 300;
    const auto getMapX = [outW](double x) { return outW/2 + x / M_PI * outW/2; };
    const auto getMapY = [outH](double y) { return outH/2 - y * outH/2; };
    {
        HtmlAnim::Vec2Vector points;
        for(int i = 0; i < sections + 1; ++i) {
            const double x = -M_PI + 2 * M_PI / sections * i;
            const auto y = sin(x);
            points.emplace_back(HtmlAnim::Vec2(getMapX(x), getMapY(y)));
        }
        anim.frame().save()
            .fill_style("white").rect(0, 0,
                static_cast<double>(anim.get_width()), static_cast<double>(anim.get_height()), true)
            .stroke_style("gray")
            .line(0,outH/2, outW,outH/2)
            .line(outW/2,0, outW/2,outH)
            .stroke_style("green")
            .line(points);
    }
    anim.add_layer();

    const size_t popSize = 1000;
    const double mutationSpread = 1;
    const double mutationHalflife = 100;
    const size_t numGens = 100;

    Population pop;
    for(size_t i = 0; i < popSize; ++i) {
        pop.addIndividual(std::make_unique<NnIndividual>());
    }

    const auto start = std::chrono::high_resolution_clock::now();

    size_t generation = 1;
    std::vector<double> outputs;
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
                      << " // avg " << sum / pop.size()
                      << " // best " << pop.getIndividual(0)->getFitness()
                      << "\n";

            const auto& nn = dynamic_cast<NnIndividual*>(pop.getIndividual(0))->nn;
            HtmlAnim::Vec2Vector points;
            for(int i = 0; i < sections + 1; ++i) {
                const double x = -M_PI + 2 * M_PI / sections * i;
                double inputs = x / M_PI;
                const auto resultIdx = nn.run(&inputs, outputs);
                const auto y = outputs[resultIdx];
                points.emplace_back(HtmlAnim::Vec2(getMapX(x), getMapY(y)));
            }
            anim.frame().save()
                .text(50, 50, std::to_string(generation))
                .stroke_style("red")
                .line(points);
            anim.next_frame();
        }

        ++generation;
    }

    const auto stop = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Time: " << duration.count() << " ms" << std::endl;

    anim.write_file("progress.html");
}

int main(int argc, char **argv)
{
    evolution1();

    return 0;
}
