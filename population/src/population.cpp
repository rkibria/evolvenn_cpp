#include "population/population.h"

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
