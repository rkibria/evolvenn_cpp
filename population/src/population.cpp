#include "population/population.h"

#include <algorithm>
#include <iostream>

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
    if(!isFirstGeneration) {
        const auto halfSize = individuals->size() / 2;
        for(size_t i = 0; i < halfSize; ++i) {
            const auto& parent = (*individuals)[i];
            const auto& offspring = (*individuals)[halfSize + i];

            offspring->mutateFrom(parent.get(), spread);
            if(i != 0) {
                parent->mutate(spread);
            }
        }
    }

    for(const auto& uptr : *individuals) {
        uptr->setFitness(0);
        uptr->evaluate();
    }

    std::sort(individuals->begin(), individuals->end(),
              [](const std::unique_ptr<Individual>& a,
              const std::unique_ptr<Individual>& b) { return a->getFitness() < b->getFitness(); } );

    isFirstGeneration = false;
}
