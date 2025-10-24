#include "matrix.h"
#include <exception>
//#include <format>
#include <iostream>

// Later: see if we stil need some of the vectors, that way we can rewrite the input and save copying

// TODO: Parallelize later
std::vector<double> operator+(const std::vector<double>& v1, const std::vector<double>& v2) 
{
    if (v1.size() != v2.size()) {
        const auto msg = "Adding vectors of sizes " + std::to_string(v1.size()) + " and "  + std::to_string(v2.size());
        throw std::invalid_argument(msg);

        //throw std::invalid_argument(std::format("Adding vectors of sizes {} and {}", v1.size(), v2.size()));
    }

    std::vector<double> res(v1.size());
    for (size_t i = 0; i < v1.size(); i++) {
        res[i] = v1[i] + v2[i];
    }
    return res;
}

// TODO: Parallelize later
double operator*(const std::vector<double>& v1, const std::vector<double>& v2)
{
    if (v1.size() != v2.size()) {
        const auto msg = "Dotting vectors of sizes " + std::to_string(v1.size()) + " and "  + std::to_string(v2.size());
        throw std::invalid_argument(msg);

        //throw std::invalid_argument(std::format("Dotting vectors of sizes {} and {}", v1.size(), v2.size()));
    }

    double res = 0.0;
    for (size_t i = 0; i < v1.size(); i++) {
        res += v1[i]*v2[i];
    }
    return res;
}

// TODO: Parallelize later
std::vector<double> operator+(const std::vector<double>& v1, double c)
{
    std::vector<double> res(v1.size());
    for (int i = 0; i < v1.size(); i++) {
        res[i] = c + v1[i];
    }
    return res;
}

std::vector<double> operator+(double c, const std::vector<double>& v2) {
    return v2 + c;
}

// TODO: Parallelize later
std::vector<double> operator*(const std::vector<double>& v1, double c)
{
    std::vector<double> res(v1.size());
    for (int i = 0; i < v1.size(); i++) {
        res[i] = c * v1[i];
    }
    return res;
}

std::vector<double> operator*(double c, const std::vector<double>& v2) {
    return v2 * c;
}

void LPP::print_object(const std::vector<double>& v)
{
    std::cout << '{';
    for (const double d : v) {
        std::cout << d << ", ";
    }
    std::cout << "}\n";
}

void LPP::print_object(const Matrix& m)
{
    m.print_entries();
}

// TODO: Parallelize later
// See if we can steal m instead of copying it
LPP::Matrix::Matrix(const std::vector<std::vector<double>>& m):
    entries{m}
{
}

// TODO: Parallelize later
// See if we can also have a random init
LPP::Matrix::Matrix(const size_t rows, const size_t cols):
    entries{rows}
{
    for (size_t r = 0; r < rows; r++) {
        entries[r] = std::vector<double>(cols, 0.0);
    }
}

// TODO: Parallelize later
LPP::Matrix::Matrix(const size_t rows, const size_t cols, const double c):
    entries{rows}
{
    for (size_t r = 0; r < rows; r++) {
        entries[r] = std::vector<double>(cols, c);
    }
}

size_t LPP::Matrix::rows() const {return entries.size();}

size_t LPP::Matrix::cols() const
{
    if (entries.empty()) return 0;

    return entries[0].size();
}

void LPP::Matrix::print_entries() const
{
    std::cout << "{\n";
    for (size_t i = 0; i < entries.size(); i++) {
        print_object(entries[i]);
    }
    std::cout << "}\n";
}

double LPP::Matrix::get(size_t i, size_t j) const
{
    if (i >= entries.size() || j >= entries[0].size()) {
        throw std::invalid_argument("Vector entry extraction out of range");
    }
    return entries[i][j];
}

void LPP::Matrix::set(size_t i, size_t j, double s)
{
    if (i >= entries.size() || j >= entries[0].size()) {
        throw std::invalid_argument("Vector entry setting out of range");
    }
    entries[i][j] = s;
}

// TODO: Parallelize later
std::vector<double> LPP::Matrix::operator*(const std::vector<double>& v) const
{
    if (entries.empty()) {
        const auto msg = "Attempting to multiply by empty matrix";
        throw std::invalid_argument(msg);
    }
    if (entries[0].size() != v.size()) {
        const auto msg = "Attempting to multiply " + std::to_string(entries.size()) + 'x' + std::to_string(entries[0].size()) + " by a 1x" + std::to_string(v.size()) + " vector";
        throw std::invalid_argument(msg);
    }

    std::vector<double> res(entries.size());
    for (size_t i = 0; i < res.size(); i++) {
        res[i] = entries[i] * v;
    }
    return res;
}

bool LPP::same_dims(const Matrix& m1, const Matrix& m2)
{
    return m1.rows() == m2.rows() && m1.cols() == m2.cols();
}