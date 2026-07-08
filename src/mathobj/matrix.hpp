#ifndef LPP_MATRIX_HPP
#define LPP_MATRIX_HPP

#include "vector_overloads.hpp"
#include "../functions/random_gen.hpp"
#include <iostream>
#include <span>

namespace LPP
{

class Matrix {
    //std::vector<std::vector<float>> entries_; // TODO: replace this with single vector and update methods to make class more cache friendly
    size_t rows_;
    size_t cols_;
    std::vector<float> entries_;

public:
    // Constructors
    Matrix(const std::vector<std::vector<float>>& m);                       // Const version
    Matrix(std::vector<std::vector<float>>& m);                             // Non-const version
    Matrix(size_t rows, size_t cols);                                       // Initialize to zeroes
    Matrix(size_t rows, size_t cols, float c);                              // Initialize to values
    Matrix(size_t rows, size_t cols, distribution::ProbabilityDistribution* pd); // Random init

    // Display information
    size_t  rows() const;
    size_t  cols() const;
    void    print_entries(std::ostream& os = std::cout) const;

    // Matrix operationss
    std::span<const float> operator[](size_t i) const;                     // Const version
    std::span<float>       operator[](size_t i);                           // Non-const versions
    std::vector<float>     operator*(const std::vector<float>& v) const;
    Matrix&                operator-=(const Matrix& m);
    Matrix&                operator*=(float c);
    float                  get(size_t i, size_t j) const;
    void                   set(size_t i, size_t j, float s);
    void                   set_all(float s);
    void                   set_row(size_t row, const std::vector<float>& v);

    // Operations used for regularization
    float sum_entries_abs() const;
    float sum_entries_sqr() const;
    float sum_entries_elastic(float a) const;
};

void print_object(const Matrix& m, std::ostream& os = std::cout);                // Display matrix
void print_object(const std::span<const float> v, std::ostream& os = std::cout); // Display vector
void print_object(const std::vector<float>& v, std::ostream& os = std::cout);    // Display vector

bool same_dims(const Matrix& m1, const Matrix& m2); // Compare two for same sizes

void matrix_sub_helper(Matrix& m1, const LPP::Matrix& m2, size_t start, size_t end); // Matrix-matrix subtraction helper
void matrix_mult_helper(Matrix& m1, size_t start, size_t end, float c);              // Matrix-scalar multiplication helper

} // namespace LPP

#endif
