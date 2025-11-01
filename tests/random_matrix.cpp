#include "../src/LPP.h"
#include <memory>

int main() {
    auto standard_norm      = LPP::STANDARD_NORMAL;
    auto scale_shift_norm   = std::make_shared<LPP::Normal>(10, 4);
    auto unif               = LPP::CENTRAL_UNIF;
    auto scale_shift_unif   = std::make_shared<LPP::Uniform>(-60, -45.5);

    LPP::Matrix m1(10, 10, standard_norm);
    LPP::Matrix m2(10, 10, scale_shift_norm);
    LPP::Matrix m3(10, 10, unif);
    LPP::Matrix m4(10, 10, scale_shift_unif);

    LPP::print_object(m1);
    LPP::print_object(m2);
    LPP::print_object(m3);
    LPP::print_object(m4);
}