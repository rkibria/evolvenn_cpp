#ifndef POPULATION_H
#define POPULATION_H

#include "population/individual.h"

#include <vector>
#include <memory>


using PopulationVector = std::vector<std::unique_ptr<Individual>>;

class Population
{
public:
    Population();
    ~Population();

    size_t size() const;
    Individual* getIndividual(size_t i) const;
    void addIndividual(std::unique_ptr<Individual>&& idv);

private:
    std::unique_ptr<PopulationVector> individuals;

};

#endif
