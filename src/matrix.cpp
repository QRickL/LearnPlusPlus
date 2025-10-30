#include "matrix.h"
#include <exception>
#include <iostream>

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

LPP::Matrix::Matrix(const std::vector<std::vector<double>>& m)
{
    entries = m;
}

LPP::Matrix::Matrix(std::vector<std::vector<double>>& m)
{
    entries = std::move(m);
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

size_t LPP::Matrix::rows() const
{
    return entries.size();
}

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

const std::vector<double>& LPP::Matrix::operator[](size_t i) const
{
    return entries[i];
}

std::vector<double>& LPP::Matrix::operator[](size_t i)
{
    return entries[i];
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

// TODO: implement
LPP::Matrix& LPP::Matrix::operator-=(const Matrix& m)
{
    // make sure to check sizes
    // amke sure to actually implement this lol

    for (size_t i = 0; i < rows(); i++) {
        entries[i] -= m.entries[i];
    }
    return *this;
}

// TODO: parallelize
LPP::Matrix& LPP::Matrix::operator*=(const double c)
{
    for (size_t i = 0; i < entries.size(); i++) {
        entries[i] *= c;
    }
    return *this;
}