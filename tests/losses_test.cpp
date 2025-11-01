#include "../src/LPP.h"
#include <iostream>

int main() {
    
    // Continuous data for MSE
    const LPP::Matrix x1({
        {1, 1, 1, 1},
        {3, 3, 3, 3},
        {5, 5, 5, 5},
        {2, 2, 2, 2},
        {6, 6, 6, 6},
        {3, 3, 3, 3}
    });

    const LPP::Matrix y1({
        {0.342, 0.13, -0.213, 1.23},
        {3.123, -0.1, 2.683, 3.64},
        {3.534, 7.234, 2.1, 5.023},
        {2.5, 1.45, 3.01, 2.234},
        {6.2, -12, 4.65, 7.52},
        {3.13, 4.1, 2.1565, 4.3341}
    });

    // Binary data for BCE
    const LPP::Matrix x2({
        {0.231}, {0.134}, {0.234}, {0.10041}, {0.565}, {0.23}, {0.9912}, {0.743}
    });

    const LPP::Matrix y2({
        {0}, {0}, {0}, {0}, {1}, {0}, {1}, {1}
    });

    // Categorical data for CE
    const LPP::Matrix x3({
        {0.6, 0.1, 0.05, 0.05, 0.1, 0.1},
        {0.07, 0.6, 0.03, 0.1, 0.15, 0.05},
        {0.025, 0.05, 0.05, 0.1, 0.75, 0.025}
    });

    const LPP::Matrix y3({
        {1, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0}
    });

    std::cout << "MSE: " << LPP::MEAN_SQUARED_ERROR->apply_itself(x1, y1)   << std::endl;
    std::cout << "BCE: " << LPP::BINARY_CROSS_ENTROPY->apply_itself(x2, y2) << std::endl;
    std::cout << "CE: "  << LPP::CROSS_ENTROPY->apply_itself(x3,y3)         << std::endl;
}