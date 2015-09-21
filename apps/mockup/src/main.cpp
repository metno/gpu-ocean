#include "manager.h"
#include <vector>
#include <iostream>

int main(int argc, char *argv[])
{
    // *** Phase 1: Initialize Manager
    Manager::init(argc, argv);
    Manager &mgr = Manager::instance();


    // *** Phase 2: Initialize a new simulation run
    mgr.simInit();


    // *** Phase 3: Run simulation
    while (mgr.simInProgress())
        mgr.simStep();


    // *** Phase 4: Process results of simulation
    std::vector<float> results = mgr.simResults();

    std::cout << "bravo!\n";

    return 0;
}
