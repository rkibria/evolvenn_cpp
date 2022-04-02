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

double getGaussianRand(double mean, double stddev)
{
    std::normal_distribution<double> dist(mean, stddev);
    return dist(generator);
}

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

constexpr int sections = 100;
double targetFunction(double x)
{
    return sin(x);
    // return x == 0 ? 0 : (0.3 * x * sin(30 / x));
}

class NnIndividual : public Individual
{
public:
    NnIndividual() : nn(1, {2, 1}, true), stddev{ 0.25 }
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
        std::uniform_real_distribution<> dis(0.8, 1.2);
        stddev *= dis(generator);
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
    const int outW = 500, outH = 300;
    HtmlAnim::HtmlAnim anim("Evolution progress", outW, 3*outH);
    anim.set_num_surfaces(1);
    anim.write_file_on_destruct("progress.html");

    const auto getMapX = [outW](double x) { return outW/2 + x / M_PI * outW/2; };
    const auto getMapY = [outH](int section, double y) { return section*outH + outH/2 - y * outH/2 * 0.9; };
    {
        HtmlAnim::Vec2Vector points;
        for(int i = 0; i < sections + 1; ++i) {
            const double x = -M_PI + 2 * M_PI / sections * i;
            const auto y = targetFunction(x);
            points.emplace_back(HtmlAnim::Vec2(getMapX(x), getMapY(0, y)));
        }
        anim.frame().save()
            .fill_style("white").rect(0, 0,
                static_cast<double>(anim.get_width()), static_cast<double>(anim.get_height()), true)
            .stroke_style("gray")
            .line(0,outH/2, outW,outH/2)
            .line(outW/2,0, outW/2,2*outH)
            .line(0,outH+outH/2, outW,outH+outH/2)
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

    const auto drawBest = [&anim, &best, &outputs, &getMapX, &getMapY, outW, outH](size_t generation, int waits) {
        HtmlAnim::Vec2Vector points, points0, points1;
        std::vector<std::vector<double>> allOutputs;
        for(int i = 0; i < sections + 1; ++i) {
            const double x = -M_PI + 2 * M_PI / sections * i;
            double inputs = x / M_PI;
            const auto resultIdx = best.nn.run(&inputs, outputs, &allOutputs);
            const auto y = outputs[resultIdx];
            const auto mapX = getMapX(x);
            points.emplace_back(HtmlAnim::Vec2(mapX, getMapY(0, y)));
            points0.emplace_back(HtmlAnim::Vec2(mapX, getMapY(1, allOutputs[0][0])));
            points1.emplace_back(HtmlAnim::Vec2(mapX, getMapY(1, allOutputs[0][1])));
        }

        constexpr auto l11color = "blue";
        constexpr auto l12color = "purple";
        const auto& weights = best.nn.getWeights();
        anim.frame()
            .surface(0)
            .save()
            .fill_style("white")
            .rect(0,outH, outW,outH, true)
            .stroke_style("gray")
            .line(outW/2,outH, outW/2,2*outH)
            .line(0,outH+outH/2, outW,outH+outH/2)
            .stroke_style(l11color)
            .line(points0)
            .stroke_style(l12color)
            .line(points1);
        anim.frame().save()
            .drawImage(0, 0,outH, outW,outH, 0,outH, outW,outH)
            .text(10, 10, std::string("Generation ") + std::to_string(generation))
            .stroke_style("red")
            .line(points)
            .fill_style(l11color)
            .text(10, 30,  std::string("L1.1: relu(") + std::to_string(weights[0]) + " + IN * " + std::to_string(weights[1]) + ")")
            .fill_style(l12color)
            .text(10, 50,  std::string("L1.2: relu(") + std::to_string(weights[2]) + " + IN * " + std::to_string(weights[3]) + ")")
            .fill_style("black")
            .text(10, 70, std::string("O1: ") + std::to_string(weights[4])
                + " + L1.1 * " + std::to_string(weights[5])
                + " + L1.2 * " + std::to_string(weights[6]))
            .wait(waits);
    };

    const size_t numGens = 100;
    int numBests = 0;
    do {
        pop.evolve();
        const auto& curBest = *(dynamic_cast<NnIndividual*>(pop.getIndividual(0)));
        if(generation == 1 || curBest.getFitness() < best.getFitness()) {
            best = curBest;
            ++numBests;
        }

        // if(generation == 1 || generation % 100 == 0) {
        //     const auto stop = std::chrono::high_resolution_clock::now();
        //     const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        //     std::cout << "gen " << generation << ": best " << best.getFitness()
        //                 << " // #improves " << numBests
        //                 << " // time " << duration.count() << " ms" 
        //                 << "\n";
        //     drawBest(generation, 180);
        //     numBests = 0;
        // }

        ++generation;
    } while(generation < numGens);

    drawBest(generation, 180);
}

int main(int argc, char **argv)
{
    generator.seed(time(nullptr));

    // converging1();
    evolution1();

    return 0;
}
