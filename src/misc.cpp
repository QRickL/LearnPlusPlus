#include "misc.h"

double dot(const std::vector<double> &vec1, const std::vector<double> &vec2) {
    // Ensure same size
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors being dotted must be of the same size");
    }

    double val = 0;

    // Calculate dot product
    for (int i = 0; i < vec1.size(); i++) val += (vec1[i] * vec2[i]);

    return val;
}
