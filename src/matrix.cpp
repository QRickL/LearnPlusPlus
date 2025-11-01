#include "matrix.h"
#include <exception>
#include <iostream>
#include <random>

LPP::Matrix::Matrix(const std::vector<std::vector<double>>& m) {entries = m;}

LPP::Matrix::Matrix(std::vector<std::vector<double>>& m) {entries = std::move(m);}

// TODO: parallelize
LPP::Matrix::Matrix(const size_t rows, const size_t cols): entries{rows}
{
    for (size_t r = 0; r < rows; r++) {
        entries[r] = std::vector<double>(cols, 0.0);
    }
}

// TODO: parallelize
LPP::Matrix::Matrix(const size_t rows, const size_t cols, const double c): entries{rows}
{
    for (size_t r = 0; r < rows; r++) {
        entries[r] = std::vector<double>(cols, c);
    }
}

// TODO: parallelize
LPP::Matrix::Matrix(const size_t rows, const size_t cols, const std::shared_ptr<LPP::ProbabilityDistribution>& pd): entries{rows}
{
    for (size_t r = 0; r < rows; r++) {
        entries[r].resize(cols);
        for (size_t c = 0; c < cols; c++) {
            entries[r][c] = pd->sample();
        }
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
    std::cout << '}' << std::endl;
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

const std::vector<double>& LPP::Matrix::operator[](size_t i) const {return entries[i];}

std::vector<double>& LPP::Matrix::operator[](size_t i) {return entries[i];}

// TODO: parallelize
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

// TODO: parallelize
LPP::Matrix& LPP::Matrix::operator-=(const Matrix& m)
{
    if (!same_dims((*this), m)) {
        const auto msg = "Attempting to subtract matrices of different sizes";
        throw std::invalid_argument(msg);
    }

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

bool LPP::same_dims(const Matrix& m1, const Matrix& m2)
{
    return m1.rows() == m2.rows() && m1.cols() == m2.cols();
}

void LPP::print_object(const std::vector<double>& v)
{
    std::cout << '{';
    for (const double d : v) {
        std::cout << d << ", ";
    }
    std::cout << '}' << std::endl;
}

void LPP::print_object(const Matrix& m) {m.print_entries();}