#ifndef MATRIX_H
#define MATRIX_H

#include "vector"
#include "exception"
#include "thread"
#include "iostream"

// ***
// TO DO: Create a custom namespace later
// ***

std::vector<double> add_vectors(const std::vector<double> &v1, const std::vector<double> &v2);

double dot_prod(const std::vector<double> &v1, const std::vector<double> &v2);

class Matrix
{
    std::vector<std::vector<double>> entry;

    void multiplication_helper(std::vector<double> &result, const std::vector<double> &x, size_t start, size_t end);

public:
    // Construct by explicitly stating entries
    // m is not to be used after initialization
    Matrix(const std::vector<std::vector<double>> &m);

    // Construct using row and column dimensions
    Matrix(const size_t row, const size_t column);

    // ***
    // TO DO: Should probably have more constructors
    // ***

    // Setting entries
    std::vector<double> &operator[](size_t row);

    // Supports matrix-vector multiplication
    std::vector<double> operator*(const std::vector<double> &x);

    size_t getRows() const;

    size_t getCols() const;

    void display_entries() const;
};

#endif