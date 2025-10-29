#ifndef LPP_MATRIX_H
#define LPP_MATRIX_H

#include <vector>

// Vector addition
std::vector<double> operator+(const std::vector<double>& v1, const std::vector<double>& v2);

// Vector dot product
double operator*(const std::vector<double>&v1, const std::vector<double>& v2);

// For adding biases
std::vector<double> operator+(const std::vector<double>& v1, double c);
std::vector<double> operator+(double c, const std::vector<double>& v2);

// Vector scalar multiplication
std::vector<double> operator*(const std::vector<double>& v1, double c);
std::vector<double> operator*(double c, const std::vector<double>& v2);

// others documentation later...
std::vector<double>& operator-=(std::vector<double>& v1, const std::vector<double>& v2);
std::vector<double>& operator*=(std::vector<double>& v1, const double c);

namespace LPP
{

class Matrix {
    std::vector<std::vector<double>> entries;

public:
    // Construct using another matrix
    Matrix(const std::vector<std::vector<double>>& m);

    // Construct based on given sizes
    Matrix(const size_t rows, const size_t cols);

    // Construct based on sizes and initial value
    Matrix(const size_t rows, const size_t cols, const double c);

    // TODO: more constructors

    // Display information
    size_t rows() const;
    size_t cols() const;
    void print_entries() const;

    // Matrix operations
    double get(size_t i, size_t j) const;
    void set(size_t i, size_t j, double s);

    std::vector<double> operator*(const std::vector<double>& v) const;
    Matrix& operator-=(const Matrix& m);
    Matrix& operator*=(const double c);

};

// Display a matrix
void print_object(const Matrix& m);

// Display a vector
void print_object(const std::vector<double>& v);

bool same_dims(const Matrix& m1, const Matrix& m2);

} // namespace LPP

#endif
