#ifndef SIMULATOR_H
#define SIMULATOR_H

class ProgramOptions;

// This class manages the low-level aspects of a simulation.
class Simulator
{
public:
    Simulator(ProgramOptions *);
    virtual ~Simulator();
    void init();
    int nextStep() const;
    int finalStep() const;
    void execNextStep();
    void printStatus() const;
private:
    ProgramOptions *options_;
    int next_step_;
    int final_step_;
};

#endif // SIMULATOR_H
