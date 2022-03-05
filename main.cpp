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

constexpr int sections = 100;

const static std::vector<std::string> CSS_COLOR_NAMES = {
    "AliceBlue","AntiqueWhite","Aqua","Aquamarine","Azure","Beige","Bisque"
    ,"BlanchedAlmond","Blue","BlueViolet","Brown","BurlyWood","CadetBlue","Chartreuse","Chocolate"
    ,"Coral","CornflowerBlue","Cornsilk","Crimson","Cyan","DarkBlue","DarkCyan","DarkGoldenRod",
    "DarkGray","DarkGrey","DarkGreen","DarkKhaki","DarkMagenta","DarkOliveGreen","Darkorange",
    "DarkOrchid","DarkRed","DarkSalmon","DarkSeaGreen","DarkSlateBlue","DarkSlateGray",
    "DarkSlateGrey","DarkTurquoise","DarkViolet","DeepPink","DeepSkyBlue","DimGray","DimGrey",
    "DodgerBlue","FireBrick","FloralWhite","ForestGreen","Fuchsia","Gainsboro","GhostWhite","Gold"
    ,"GoldenRod","Gray","Grey","Green","GreenYellow","HoneyDew","HotPink","IndianRed","Indigo",
    "Ivory","Khaki","Lavender","LavenderBlush","LawnGreen","LemonChiffon","LightBlue","LightCoral"
    ,"LightCyan","LightGoldenRodYellow","LightGray","LightGrey","LightGreen","LightPink",
    "LightSalmon","LightSeaGreen","LightSkyBlue","LightSlateGray","LightSlateGrey",
    "LightSteelBlue","LightYellow","Lime","LimeGreen","Linen","Magenta","Maroon",
    "MediumAquaMarine","MediumBlue","MediumOrchid","MediumPurple","MediumSeaGreen",
    "MediumSlateBlue","MediumSpringGreen","MediumTurquoise","MediumVioletRed","MidnightBlue",
    "MintCream","MistyRose","Moccasin","NavajoWhite","Navy","OldLace","Olive","OliveDrab","Orange"
    ,"OrangeRed","Orchid","PaleGoldenRod","PaleGreen","PaleTurquoise","PaleVioletRed","PapayaWhip"
    ,"PeachPuff","Peru","Pink","Plum","PowderBlue","Purple","Red","RosyBrown","RoyalBlue",
    "SaddleBrown","Salmon","SandyBrown","SeaGreen","SeaShell","Sienna","Silver","SkyBlue",
    "SlateBlue","SlateGray","SlateGrey","Snow","SpringGreen","SteelBlue","Tan","Teal","Thistle",
    "Tomato","Turquoise","Violet","Wheat","White","WhiteSmoke","Yellow","YellowGreen" };

const std::string& getColorName(int i)
{
    return CSS_COLOR_NAMES[static_cast<size_t>(i) % CSS_COLOR_NAMES.size()];
}

double getGaussianRand0(double mean, double stddev)
{
    std::normal_distribution<double> dist(mean, stddev);
    return dist(generator);
}

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

    NnIndividual() : nn(1, {8, 8, 1}, true)
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

void converging1()
{
    const int outW = 500;
    const auto getMapX = [outW](double x) { return outW/2 + x * outW/2; };
    const auto getMapY = [outW](double y) { return outW/2 - y * outW/2; };

    HtmlAnim::HtmlAnim anim("converging", outW, outW);
    anim.frame().save()
        .add_drawable(HtmlAnimShapes::subdivided_grid(0,0, outW/2,outW/2, 2,2, 2,2))
        .line_width(3)
        .line(0,outW/2, outW,outW/2)
        .line(outW/2,0, outW/2,outW);
    anim.add_layer();

    NeuralNet nn(1, {1}, true);

    const auto initNn = [&nn]() {
        auto& weights = nn.getWeights();
        for(auto& w : weights) {
            w = getGaussianRand0(0, 1);
        }
    };

    const auto mutateNn = [&nn](double stddev) {
        auto& weights = nn.getWeights();
        for(auto& w : weights) {
            w = getGaussianRand0(0, stddev);
        }
    };

    initNn();

    std::vector<double> outputs;
    const double radius = 1.0;
    const int sections = 2;
    for(int k = 0; k < 500; ++k) {

        mutateNn(1);

        HtmlAnim::Vec2Vector points;
        for(int i = 0; i < sections + 1; ++i) {
            const double x = -radius + 2 * radius / sections * i;
            const double inputs = x / radius; // map input to -1, 1 range
            const auto resultIdx = nn.run(&inputs, outputs);
            const auto actual = outputs[resultIdx];
            points.emplace_back(HtmlAnim::Vec2(getMapX(x), getMapY(actual)));
        }
        anim.frame().save()
            .stroke_style(getColorName(k))
            .line(points);
    }

    anim.write_file("first_gen_nets.html");
}

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
    NnIndividual best;
    std::vector<double> outputs;
    while(generation < numGens) {
        const double halflifeFactor = 1.0; // TODO this returns inf (mutationHalflife > 0) ? pow(2.0, -(generation % 1000) / mutationHalflife) : 1;
        const double spread = (1.0 - (generation / static_cast<double>(numGens))) * 0.25; // TODO std::max(mutationSpread * halflifeFactor, mutationSpread * 0.01);

        pop.evolve(spread);

        const auto stop = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        double sum = 0;
        for(size_t i = 0; i < pop.size(); ++i) {
            sum += pop.getIndividual(i)->getFitness();
        }

        const auto& curBest = *(dynamic_cast<NnIndividual*>(pop.getIndividual(0)));
        std::cout << "gen " << generation << ": best " << curBest.getFitness()
                    << " // spread " << spread
                    << " // avg " << sum / pop.size()
                    << " // time " << duration.count() << " ms" 
                    << "\n";

        if(generation == 1 || generation == numGens-1 || curBest.getFitness() < best.getFitness()) {
            std::cout << "drawing\n";
            best = curBest;

            HtmlAnim::Vec2Vector points;
            for(int i = 0; i < sections + 1; ++i) {
                const double x = -M_PI + 2 * M_PI / sections * i;
                double inputs = x / M_PI;
                const auto resultIdx = curBest.nn.run(&inputs, outputs);
                const auto y = outputs[resultIdx];
                points.emplace_back(HtmlAnim::Vec2(getMapX(x), getMapY(y)));
            }
            anim.frame().save()
                .text(10, 10, std::string("Generation ") + std::to_string(generation))
                .stroke_style("red")
                .line(points)
                .wait((generation == numGens-1) ? 180 : 20);
            anim.next_frame();
        }

        ++generation;
    }

    anim.write_file("progress.html");
}

int main(int argc, char **argv)
{
    // evolution1();

    converging1();

    return 0;
}
