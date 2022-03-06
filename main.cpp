#include <iostream>
#include <cmath>
#include <chrono>
#include <random>
#include <iomanip>
#include <cassert>

#include "neuralnet/neuralnet.h"
#include "population/population.h"

#include "htmlanim_shapes.hpp"

double targetFunction(double x)
{
    return sin(x);
    // return x == 0 ? 0 : (0.3 * x * sin(30 / x));
}

std::default_random_engine generator;

std::uniform_int_distribution<> coinDistrib(0, 1);

double getGaussianRand(double mean, double stddev)
{
    std::normal_distribution<double> dist(mean, stddev);
    return dist(generator);
}

int getCointoss()
{
    return coinDistrib(generator);
}

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

const double startStddev = 0.25;
class NnIndividual : public Individual
{
public:
    NnIndividual() : nn(1, {8, 8, 1}, true), stddev{ startStddev }
    {
        auto& weights = nn.getWeights();
        for(auto& w : weights) {
            w = getGaussianRand(0, 1.0);
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
            const auto expect = targetFunction(x);

            double inputs = x / M_PI;

            const auto resultIdx = nn.run(&inputs, outputs);

            const auto actual = outputs[resultIdx];

            auto diff = fabs(actual - expect);
            const auto diffsq = diff * diff;
            fitness += diffsq;
        }
    }

    void mutate() override
    {
        auto& weights = nn.getWeights();
        for(auto& w : weights) {
            const auto variation = getGaussianRand(0, stddev);
            w += variation;
        }
        stddev *= getCointoss() ? 0.8 : 1.2;
        stddev = std::max(0.001, stddev);
    }

    void mutateFrom(const Individual* other) override
    {
        const auto otherNn = dynamic_cast<const NnIndividual*>(other);
        assert(this->nn.getInputs() == otherNn->nn.getInputs());
        assert(this->nn.getLayerSizes() == otherNn->nn.getLayerSizes());

        stddev = otherNn->stddev;
        nn = otherNn->nn;
        mutate();
    }

    void dump(std::ostream& os) const override
    {
        os << nn.getWeights()[0] << "/" << nn.getWeights()[1] << "/" << nn.getWeights()[2];
    }

    NeuralNet nn;
    double stddev{0};
};

void converging1()
{
    const int outW = 500;
    const auto getMapX = [outW](double x) { return outW/2 + x * outW/2; };
    const auto getMapY = [outW](double y) { return outW/2 - y * outW/2; };

    HtmlAnim::HtmlAnim anim("converging", outW, outW);
    anim.frame().save()
        .fill_style("black").rect(0, 0, outW, outW, true)
        .add_drawable(HtmlAnimShapes::subdivided_grid(0,0, outW/2,outW/2, 2,2, 2,2))
        .stroke_style("white")
        .line_width(3)
        .line(0,outW/2, outW,outW/2)
        .line(outW/2,0, outW/2,outW);
    anim.add_layer();

    NeuralNet nn(1, {1}, false);

    const auto initNn = [&nn]() {
        auto& weights = nn.getWeights();
        for(auto& w : weights) {
            w = getGaussianRand(0, 1);
        }
    };

    const auto mutateNn = [&nn](double stddev) {
        auto& weights = nn.getWeights();
        for(auto& w : weights) {
            w = getGaussianRand(0, stddev);
        }
    };


    // initNn();

    std::vector<double> outputs;
    const double radius = 1.0;
    const int sections = 100;
    for(int k = 0; k < 10; ++k) {

        initNn();
        // mutateNn(1);

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
    HtmlAnim::HtmlAnim anim("Evolution progress");
    const int outW = 500, outH = 300;
    const auto getMapX = [outW](double x) { return outW/2 + x / M_PI * outW/2; };
    const auto getMapY = [outH](double y) { return outH/2 - y * outH/2 * 0.9; };
    {
        HtmlAnim::Vec2Vector points;
        for(int i = 0; i < sections + 1; ++i) {
            const double x = -M_PI + 2 * M_PI / sections * i;
            const auto y = targetFunction(x);
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

    Population pop;
    const size_t popSize = 1000;
    for(size_t i = 0; i < popSize; ++i) {
        pop.addIndividual(std::make_unique<NnIndividual>());
    }

    const auto start = std::chrono::high_resolution_clock::now();

    size_t generation = 1;
    NnIndividual best;
    std::vector<double> outputs;

    const auto drawBest = [&anim, &best, &outputs, &getMapX, &getMapY](size_t generation, int waits) {
        HtmlAnim::Vec2Vector points;
        for(int i = 0; i < sections + 1; ++i) {
            const double x = -M_PI + 2 * M_PI / sections * i;
            double inputs = x / M_PI;
            const auto resultIdx = best.nn.run(&inputs, outputs);
            const auto y = outputs[resultIdx];
            points.emplace_back(HtmlAnim::Vec2(getMapX(x), getMapY(y)));
        }
        anim.frame().save()
            .text(10, 10, std::string("Generation ") + std::to_string(generation))
            .stroke_style("red")
            .line(points)
            .wait(waits);
        anim.next_frame();
    };

    const size_t numGens = 2000;
    int numBests = 0;
    do {
        pop.evolve();
        const auto& curBest = *(dynamic_cast<NnIndividual*>(pop.getIndividual(0)));
        if(generation == 1 || curBest.getFitness() < best.getFitness()) {
            best = curBest;
            ++numBests;
        }

        if(generation % 100 == 0) {
            const auto stop = std::chrono::high_resolution_clock::now();
            const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            std::cout << "gen " << generation << ": best " << best.getFitness()
                        << " // #improves " << numBests
                        << " // time " << duration.count() << " ms" 
                        << "\n";
            drawBest(generation, 10);
            numBests = 0;
        }

        ++generation;
    } while(best.getFitness() > 0.01 && generation < numGens);

    drawBest(generation, 180);

    anim.write_file("progress.html");
}

int main(int argc, char **argv)
{
    generator.seed(time(nullptr));

    // converging1();
    evolution1();

    return 0;
}
