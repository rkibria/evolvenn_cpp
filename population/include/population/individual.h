#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include <iostream>

class Individual
{
public:
    virtual ~Individual() {}

    double getFitness() const { return fitness; }
    void setFitness(double f) { fitness = f; }

    virtual void evaluate() = 0;

    virtual void mutate(double spread) = 0;
    virtual void mutateFrom(const Individual*, double spread) = 0;

    virtual void dump(std::ostream& os) const {}

protected:
    double fitness{ 0 };
};

#endif
