#include "matrix.hpp"
#include "../checking/check.hpp"
//#include "parallel/matrix_parallel.h"
#include <exception>
#include <iostream>
#include <random>

LPP::Matrix::Matrix(const std::vector<std::vector<float>>& m) {entries_ = m;}

LPP::Matrix::Matrix(std::vector<std::vector<float>>& m) {entries_ = std::move(m);}

LPP::Matrix::Matrix(size_t rows, size_t cols): entries_{rows}
{
    for (size_t r = 0; r < rows; r++) {
        entries_[r] = std::vector<float>(cols, 0.0);
    }
    // if (rows < LPP::CONSTANTS::MATRIX_PARALLEL_THRESHOLD) {
    //     for (size_t r = 0; r < rows; r++) {
    //         entries_[r] = std::vector<float>(cols, 0.0);
    //     }
    // } else {
    //     LPP::matrix_parallel_init(entries_, rows, cols, 0.0);
    // }
}

LPP::Matrix::Matrix(size_t rows, size_t cols, float c): entries_{rows}
{
    for (size_t r = 0; r < rows; r++) {
        entries_[r] = std::vector<float>(cols, c);
    }
    // if (rows < LPP::CONSTANTS::MATRIX_PARALLEL_THRESHOLD) {
    //     for (size_t r = 0; r < rows; r++) {
    //         entries_[r] = std::vector<float>(cols, c);
    //     }
    // } else {
    //     LPP::matrix_parallel_init(entries_, rows, cols, c);
    // }
}

LPP::Matrix::Matrix(size_t rows, size_t cols, const std::shared_ptr<LPP::distribution::ProbabilityDistribution>& pd): entries_{rows}
{
    for (size_t r = 0; r < rows; r++) {
        entries_[r].resize(cols);
        for (size_t c = 0; c < cols; c++) {
            entries_[r][c] = pd->sample();
        }
    }
}

size_t LPP::Matrix::rows() const
{
    return entries_.size();
}

size_t LPP::Matrix::cols() const
{
    if (entries_.empty()) return 0;
    return entries_[0].size();
}

void LPP::Matrix::print_entries() const
{
    std::cout << "{\n";
    for (size_t i = 0; i < entries_.size(); i++) {
        print_object(entries_[i]);
    }
    std::cout << '}' << std::endl;
}

float LPP::Matrix::get(size_t i, size_t j) const
{
    __lpp_check__(i < entries_.size() && j < entries_[0].size(), "Matrix::get - Vector entry extraction out of range");
    
    return entries_[i][j];
}

void LPP::Matrix::set(size_t i, size_t j, float s)
{
    __lpp_check__(i < entries_.size() && j < entries_[0].size(), "Matrix::set - Vector entry setting out of range");
 
    entries_[i][j] = s;
}

const std::vector<float>& LPP::Matrix::operator[](size_t i) const {return entries_[i];}

std::vector<float>& LPP::Matrix::operator[](size_t i) {return entries_[i];}

std::vector<float> LPP::Matrix::operator*(const std::vector<float>& v) const
{
    __lpp_check__(
        !entries_.empty(),
        "Matrix::operator* - Attempting to multiply by empty matrix"
    );
    __lpp_check__(
        entries_[0].size() == v.size(),
        "Matrix::operator* - Attempting to multiply " + std::to_string(entries_.size()) + 'x' + std::to_string(entries_[0].size()) + " by a 1x" + std::to_string(v.size()) + " vector"
    );
    
    std::vector<float> res(entries_.size());
    for (size_t i = 0; i < res.size(); i++) {
        res[i] = entries_[i] * v;
    }
    return res;
}

void LPP::matrix_sub_helper(Matrix& m1, const LPP::Matrix& m2, size_t start, size_t end) {
    for (size_t r = start; r < end; r++) {
        m1[r] -= m2[r];
    }
}

LPP::Matrix& LPP::Matrix::operator-=(const Matrix& m)
{
    __lpp_check__(same_dims(*this, m), "Matrix:operator-= - Attempting to subtract matrices of different sizes");

    for (size_t i = 0; i < rows(); i++) {
        entries_[i] -= m.entries_[i];
    }
    return *this;
    
    // if (this->rows() < LPP::CONSTANTS::MATRIX_PARALLEL_THRESHOLD) {
    //     for (size_t i = 0; i < rows(); i++) {
    //         entries_[i] -= m.entries_[i];
    //     }
    // } else {
    //     std::vector<std::thread> blocks;
    //     size_t block_size = this->rows() / LPP::CONSTANTS::USE_THREADS;

    //     for (size_t r = 0; r <= LPP::CONSTANTS::USE_THREADS; r++) {
    //         size_t start  = r * block_size;
    //         size_t end    = std::min( (r + 1) * block_size, this->rows() );

    //         blocks.emplace_back(LPP::matrix_sub_helper, std::ref(*this), std::ref(m), start, end);
    //     }
    //     for (auto& block : blocks) block.join();
    // }
    // return *this;
}

void LPP::matrix_mult_helper(Matrix& m1, size_t start, size_t end, float c)
{
    for (size_t r = start; r < end; r++) {
        m1[r] *= c;
    }
}

LPP::Matrix& LPP::Matrix::operator*=(float c)
{
    for (size_t i = 0; i < entries_.size(); i++) {
        entries_[i] *= c;
    }
    return *this;

    // if (this->rows() < LPP::CONSTANTS::MATRIX_PARALLEL_THRESHOLD) {
    //     for (size_t i = 0; i < entries_.size(); i++) {
    //         entries_[i] *= c;
    //     }
    // } else {
    //     std::vector<std::thread> blocks;
    //     size_t block_size = this->rows() / LPP::CONSTANTS::USE_THREADS;

    //     for (size_t r = 0; r <= LPP::CONSTANTS::USE_THREADS; r++) {
    //         size_t start  = r * block_size;
    //         size_t end    = std::min( (r + 1) * block_size, this->rows() );

    //         blocks.emplace_back(LPP::matrix_mult_helper, std::ref(*this), start, end, c);
    //     }
    //     for (auto& block : blocks) block.join();
    // }
    // return *this;
}

bool LPP::same_dims(const Matrix& m1, const Matrix& m2)
{
    return m1.rows() == m2.rows() && m1.cols() == m2.cols();
}

void LPP::print_object(const std::vector<float>& v)
{
    std::cout << '{';
    for (float d : v) {
        std::cout << d << ", ";
    }
    std::cout << '}' << std::endl;
}

void LPP::print_object(const Matrix& m) {m.print_entries();}

float LPP::Matrix::sum_entries_abs() const
{
    float res = 0.f;
    for (size_t i = 0; i < rows(); i++) {
        for (size_t j = 0; j < cols(); j++) {
            res += std::abs(entries_[i][j]);
        }
    }
    return res;
}

float LPP::Matrix::sum_entries_sqr() const
{
    float res = 0.f;
    for (size_t i = 0; i < rows(); i++) {
        for (size_t j = 0; j < cols(); j++) {
            res += std::pow(entries_[i][j], 2);
        }
    }
    return res;
}

float LPP::Matrix::sum_entries_elastic(float a) const
{
    float res = 0.f;
    for (size_t i = 0; i < rows(); i++) {
        for (size_t j = 0; j < cols(); j++) {
            res += a * std::abs(entries_[i][j]) + (1-a) * std::pow(entries_[i][j], 2);
        }
    }
    return res;
}