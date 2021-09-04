#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

class Individual
{
public:
    virtual ~Individual() {}

    double getFitness() const { return fitness; }
    void setFitness(double f) { fitness = f; }

    virtual void mutate(double spread) = 0;
    virtual void evaluate() = 0;

private:
    double fitness{ 0 };
};

#endif
