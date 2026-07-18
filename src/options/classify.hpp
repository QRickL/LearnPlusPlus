#ifndef LPP_CLASSIFY_HPP
#define LPP_CLASSIFY_HPP

#include <span>
#include "../mathobj/matrix.hpp"

namespace LPP {
namespace classify {

size_t to_label_idx(std::span<const float> x);

float compute_accuracy(const LPP::Matrix& actual, const LPP::Matrix& estimated);

}
}

#endif
