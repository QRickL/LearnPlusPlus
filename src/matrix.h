#pragma once

#include "lpp_vec_overloads.h"
#include "functions/random_gen.h"

namespace LPP
{

template <typename number>
class Matrix {
    Vect<number> entries__;
    size_t rows__;
    size_t cols__;

public:
    // Constructors

    Matrix(const std::vector<std::vector<float>>& m);                      // Initialize from matrix of doubles. Const version. Assumes m is non-empty and all rows are same size
    Matrix(std::vector<std::vector<float>>& m);                            // Initialize from matrix of doubles. Non-const version. Assumes m is non-empty and all rows are same size
    Matrix(size_t rows, size_t cols);                                      // Initialize to zeroes
    Matrix(size_t rows, size_t cols, float initVal);                       // Initialize to given value
    Matrix(size_t rows, size_t cols, const std::shared_ptr<ProbabilityDistribution>& pd); // Initialize with random numbers from distributions

    // Information

    size_t rows() const;
    size_t cols() const;
    void   printEntries(std::ostream& os = std::cout) const; // Prints all entries of the matrix

    // Matrix operations

    // const std::vector<double>&  operator[](size_t i) const;                     // Const version
    // std::vector<double>&        operator[](size_t i);                           // Non-const versions

    Vect<number> operator*(const Vect<number>& v) const;  // Matrix vector multiplication for computing pre-activation layer outputs
    Matrix&      operator-=(const Matrix& m);             // Matrix subtraction for updating weights
    Matrix&      operator*=(number c);                    // Matrix scalar multiplication for scaling by learning rate

    const number& get(size_t i, size_t j) const;          // Getter method
    void  set(size_t i, size_t j, const number& s);       // Setter method. Copies data
    void  set(size_t i, size_t j, number&& s);            // Setter method. Steals data

};

// Compare two for same sizes
template <typename number>
bool sameDims(const Matrix<number>& m1, const Matrix<number>& m2);


// // For help in parallel computations
// void matrix_sub_helper(Matrix& m1, const LPP::Matrix& m2, const size_t start, const size_t end);
// void matrix_mult_helper(Matrix& m1, const size_t start, const size_t end, const double c);

} // namespace LPP
