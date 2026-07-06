#include "../src/LPP.hpp"
#include <memory>

int main() {
    auto standard_norm      = LPP::distribution::standard_normal;
    auto scale_shift_norm   = LPP::distribution::Normal(10, 4);
    auto unif               = LPP::distribution::standard_uniform;
    auto scale_shift_unif   = LPP::distribution::Uniform(-60, -45.5);

    LPP::Matrix m1(10, 10, &standard_norm);
    LPP::Matrix m2(10, 10, &scale_shift_norm);
    LPP::Matrix m3(10, 10, &unif);
    LPP::Matrix m4(10, 10, &scale_shift_unif);

    LPP::print_object(m1);
    LPP::print_object(m2);
    LPP::print_object(m3);
    LPP::print_object(m4);
}