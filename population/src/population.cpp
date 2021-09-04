#include "population/population.h"

#include <algorithm>

Population::Population()
    : individuals{ std::make_unique<PopulationVector>() }
{
}

Population::~Population()
{
}

Individual* Population::getIndividual(size_t i) const
{
    return (*individuals)[i].get();
}

size_t Population::size() const
{
    return individuals->size();
}

void Population::addIndividual(std::unique_ptr<Individual>&& idv)
{
    individuals->emplace_back(std::move(idv));
}

void Population::evolve(double spread)
{
    for(const auto& uptr : *individuals) {
        uptr->setFitness(0);
        uptr->evaluate();
    }

    std::sort(individuals->begin(), individuals->end(),
              [](const std::unique_ptr<Individual>& idv1,
              const std::unique_ptr<Individual>& idv2) { return idv1->getFitness() < idv2->getFitness(); } );

    const auto halfSize = individuals->size() / 2;

    for(size_t i = 0; i < halfSize; ++i) {
        const auto& parent = (*individuals)[i];
        const auto& offspring = (*individuals)[halfSize + i];

        *offspring = *parent;
        offspring->mutate(spread);
        if(i != 0) {
            parent->mutate(spread);
        }
    }

    isFirstGeneration = false;
}
