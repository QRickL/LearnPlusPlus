#ifndef LPP_MATRIX_H
#define LPP_MATRIX_H

#include "vector_overloads.hpp"
#include "../functions/random_gen.hpp"

namespace LPP
{

class Matrix {
    std::vector<std::vector<float>> entries; // TODO: replace this with single vector and update methods to make class more cache friendly

public:
    // Constructors
    Matrix(const std::vector<std::vector<float>>& m);                       // Const version
    Matrix(std::vector<std::vector<float>>& m);                             // Non-const version
    Matrix(size_t rows, size_t cols);                                       // Initialize to zeroes
    Matrix(size_t rows, size_t cols, float c);                              // Initialize to values
    Matrix(size_t rows, size_t cols, const std::shared_ptr<ProbabilityDistribution>& pd); // Random init

    // Display information
    size_t  rows() const;
    size_t  cols() const;
    void    print_entries() const;

    // Matrix operations
    const std::vector<float>&  operator[](size_t i) const;                     // Const version
    std::vector<float>&        operator[](size_t i);                           // Non-const versions
    std::vector<float>         operator*(const std::vector<float>& v) const;
    Matrix&                    operator-=(const Matrix& m);
    Matrix&                    operator*=(float c);
    float                      get(size_t i, size_t j) const;
    void                       set(size_t i, size_t j, float s);

};

void print_object(const Matrix& m);             // Display matrix
void print_object(const std::vector<float>& v); // Display vector

bool same_dims(const Matrix& m1, const Matrix& m2); // Compare two for same sizes

void matrix_sub_helper(Matrix& m1, const LPP::Matrix& m2, size_t start, size_t end); // Matrix-matrix subtraction helper
void matrix_mult_helper(Matrix& m1, size_t start, size_t end, float c);              // Matrix-scalar multiplication helper

} // namespace LPP

#endif
