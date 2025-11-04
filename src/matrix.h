#ifndef LPP_MATRIX_H
#define LPP_MATRIX_H

#include "vector_overloads.h"
#include "functions/random_gen.h"

namespace LPP
{

class Matrix {
    std::vector<std::vector<double>> entries;

public:
    // Constructors
    Matrix(const std::vector<std::vector<double>>& m);                      // Const version
    Matrix(std::vector<std::vector<double>>& m);                            // Non-const version
    Matrix(const size_t rows, const size_t cols);                           // Initialize to zeroes
    Matrix(const size_t rows, const size_t cols, const double c);           // Initialize to values
    Matrix(const size_t rows, const size_t cols, const std::shared_ptr<ProbabilityDistribution>& pd); // Random init

    // Display information
    size_t  rows() const;
    size_t  cols() const;
    void    print_entries() const;

    // Matrix operations
    const std::vector<double>&  operator[](size_t i) const;                     // Const version
    std::vector<double>&        operator[](size_t i);                           // Non-const versions
    std::vector<double>         operator*(const std::vector<double>& v) const;
    Matrix&     operator-=(const Matrix& m);
    Matrix&     operator*=(const double c);
    double      get(size_t i, size_t j) const;
    void        set(size_t i, size_t j, double s);

};

// Display matrix or vector
void print_object(const Matrix& m);
void print_object(const std::vector<double>& v);

// Compare two for same sizes
bool same_dims(const Matrix& m1, const Matrix& m2);

// For help in parallel computations
void matrix_sub_helper(Matrix& m1, const LPP::Matrix& m2, const size_t start, const size_t end);
void matrix_mult_helper(Matrix& m1, const size_t start, const size_t end, const double c);

} // namespace LPP

#endif
